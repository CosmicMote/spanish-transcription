import json
import os
import re
import sqlite3
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import whisper
import yt_dlp
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DB_PATH = Path(os.getenv("DB_PATH", str(Path(__file__).parent.parent / "transcriptions.db")))
# Uploaded YouTube login cookies live next to the DB (i.e. in /data under Docker).
YOUTUBE_COOKIES_PATH = DB_PATH.parent / "youtube_cookies.txt"
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
BATCH_SIZE = 25

# Optional path prefix for running behind a reverse proxy that forwards the
# full path unstripped (e.g. mounted at /spanish-transcription). Leave unset
# to run at the root path, which is the default behavior.
BASE_PATH = os.getenv("BASE_PATH", "").rstrip("/")
if BASE_PATH and not BASE_PATH.startswith("/"):
    BASE_PATH = "/" + BASE_PATH

app = FastAPI()
router = APIRouter(prefix=BASE_PATH)

print(f"Loading Whisper model '{WHISPER_MODEL}'...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded.")

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
# `or` (not a getenv default) so an empty value, e.g. from docker compose's
# ${VAR:-}, also falls back.
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
# Used when the Anthropic models API can't be reached (e.g. no API key yet).
FALLBACK_ANTHROPIC_MODELS = [
    {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5"},
    {"id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5"},
    {"id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5"},
    {"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5"},
]
# Local models can be slow, especially on first load.
OLLAMA_TIMEOUT_SECS = 600
# For the settings dialog's "Test connection"; allows for a cold model load.
OLLAMA_TEST_TIMEOUT_SECS = 120

# Created lazily so the app can start without an API key when using Ollama.
_claude: anthropic.Anthropic | None = None


def get_claude() -> anthropic.Anthropic:
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic()
    return _claude


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            transcribed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            timestamp TEXT NOT NULL,
            spanish TEXT NOT NULL,
            english TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS word_list (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL UNIQUE,
            part_of_speech TEXT DEFAULT '',
            definition TEXT DEFAULT '',
            example_es TEXT DEFAULT '',
            example_en TEXT DEFAULT '',
            added_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS word_cache (
            word TEXT PRIMARY KEY,
            part_of_speech TEXT DEFAULT '',
            definition TEXT DEFAULT '',
            example_es TEXT DEFAULT '',
            example_en TEXT DEFAULT '',
            cached_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS quiz_cache (
            session_id INTEGER PRIMARY KEY REFERENCES sessions(id),
            questions TEXT NOT NULL,
            cached_at TEXT NOT NULL
        );
        -- Whisper output whose translation/grading failed (e.g. AI provider
        -- unreachable), kept so it can be resumed without re-transcribing.
        CREATE TABLE IF NOT EXISTS pending_transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            source_url TEXT,
            segments TEXT NOT NULL,
            translations TEXT NOT NULL,
            elapsed_secs REAL NOT NULL,
            error TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "cefr_level" not in existing:
        conn.execute("ALTER TABLE sessions ADD COLUMN cefr_level TEXT")
    if "cefr_rationale" not in existing:
        conn.execute("ALTER TABLE sessions ADD COLUMN cefr_rationale TEXT")
    if "audio_duration_secs" not in existing:
        conn.execute("ALTER TABLE sessions ADD COLUMN audio_duration_secs REAL")
    if "transcription_secs" not in existing:
        conn.execute("ALTER TABLE sessions ADD COLUMN transcription_secs REAL")
    if "source_url" not in existing:
        conn.execute("ALTER TABLE sessions ADD COLUMN source_url TEXT")
    conn.commit()
    conn.close()


init_db()


# ── AI provider settings / completion ────────────────────────────────────────

SETTINGS_DEFAULTS = {
    "provider": "anthropic",
    "anthropic_model": DEFAULT_ANTHROPIC_MODEL,
    "ollama_host": DEFAULT_OLLAMA_HOST,
    "ollama_model": "",
}


def get_settings() -> dict:
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    stored = {r["key"]: r["value"] for r in rows}
    return {k: stored.get(k, v) for k, v in SETTINGS_DEFAULTS.items()}


def normalize_ollama_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host:
        raise HTTPException(status_code=400, detail="Ollama host is required.")
    if not re.match(r"^https?://", host):
        host = "http://" + host
    return host


def ollama_request(host: str, path: str, payload: dict | None = None, timeout: float = 10) -> dict:
    url = normalize_ollama_host(host) + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body = json.loads(body).get("error", body)
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"Ollama error ({e.code}): {body}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama at {url}: {reason}")


def complete(prompt: str, settings: dict | None = None, ollama_timeout: float = OLLAMA_TIMEOUT_SECS) -> str:
    """Send a single-turn prompt to the AI provider and return its text reply.

    Uses the saved settings unless `settings` is given (e.g. to test unsaved ones).
    """
    settings = settings or get_settings()
    if settings["provider"] == "ollama":
        if not settings["ollama_model"]:
            raise HTTPException(status_code=400, detail="No Ollama model configured. Open AI settings to choose one.")
        result = ollama_request(
            settings["ollama_host"],
            "/api/chat",
            {
                "model": settings["ollama_model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=ollama_timeout,
        )
        text = result.get("message", {}).get("content", "")
        # Some reasoning models inline their thinking in the reply.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    try:
        message = get_claude().messages.create(
            model=settings["anthropic_model"],
            # Generous cap: newer Claude models think by default and thinking
            # tokens count toward max_tokens. Billing is per token actually used.
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e}")
    except anthropic.AnthropicError as e:
        raise HTTPException(status_code=500, detail=f"Anthropic client error: {e}")
    except TypeError as e:
        # The SDK raises TypeError at request time when no API key is configured.
        if "authentication" not in str(e):
            raise
        raise HTTPException(status_code=500, detail="No Anthropic API key configured. Set the ANTHROPIC_API_KEY environment variable.")
    # Skip thinking blocks; only text blocks hold the answer.
    return "".join(b.text for b in message.content if b.type == "text").strip()


def strip_code_fences(raw: str) -> str:
    return re.sub(r"^```[a-z]*\n?", "", raw.strip()).rstrip("`").strip()


def save_session(filename: str, segments: list[dict], cefr_level: str = "", cefr_rationale: str = "",
                 audio_duration_secs: float | None = None, transcription_secs: float | None = None,
                 source_url: str | None = None) -> int:
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO sessions (filename, transcribed_at, cefr_level, cefr_rationale,
                                 audio_duration_secs, transcription_secs, source_url)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (filename, now, cefr_level, cefr_rationale, audio_duration_secs, transcription_secs, source_url),
    )
    session_id = cursor.lastrowid
    conn.executemany(
        "INSERT INTO segments (session_id, timestamp, spanish, english) VALUES (?, ?, ?, ?)",
        [(session_id, s["timestamp"], s["spanish"], s["english"]) for s in segments],
    )
    conn.commit()
    conn.close()
    return session_id


def format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def translate_batch(spanish_texts: list[str]) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(spanish_texts))
    response_text = complete(
        "Translate each numbered Spanish phrase to English. "
        "Return only the numbered translations in the same format "
        '(e.g., "1. The translation"). Do not add explanations.\n\n'
        + numbered
    )
    translations = {}
    for line in response_text.splitlines():
        match = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if match:
            idx = int(match.group(1)) - 1
            translations[idx] = match.group(2)
    return [translations.get(i, "") for i in range(len(spanish_texts))]


def estimate_cefr(spanish_texts: list[str]) -> tuple[str, str]:
    sample = "\n".join(spanish_texts[:40])[:3000]
    raw = strip_code_fences(complete(
        "Estimate the CEFR difficulty level (A1, A2, B1, B2, C1, or C2) "
        "of the following Spanish transcript for a language learner. "
        "Consider vocabulary range, sentence complexity, and topic familiarity. "
        'Reply with JSON only: {"level": "B1", "rationale": "one sentence"}. '
        "No markdown, no extra text.\n\n" + sample
    ))
    try:
        result = json.loads(raw)
        return result.get("level", "?"), result.get("rationale", "")
    except Exception as e:
        print(f"estimate_cefr parse error: {e!r} — raw response: {raw!r}")
        return "?", ""


def process_audio(audio_path: str, display_name: str, source_url: str | None = None) -> dict:
    """Transcribe, translate, grade, and save an audio file as a new session."""
    processing_start = time.time()
    result = whisper_model.transcribe(audio_path, language="es", task="transcribe")
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]

    if not segments:
        return {"segments": [], "session_id": None}

    job = {
        "filename": display_name,
        "source_url": source_url,
        "segments": segments,
        "translations": [],
        "elapsed_secs": time.time() - processing_start,
    }
    return finish_transcription(job)


def finish_transcription(job: dict, pending_id: int | None = None) -> dict:
    """Translate, grade, and save already-transcribed segments as a new session.

    If an AI step fails, the job (including any batches already translated) is
    stored as a pending transcription and the error carries its id, so it can
    be resumed later without downloading or transcribing again.
    """
    texts = [seg["text"] for seg in job["segments"]]
    try:
        # elapsed_secs only counts work that succeeded, so failed attempts
        # (which may sit in a long timeout) don't skew transcription estimates.
        while len(job["translations"]) < len(texts):
            step_start = time.time()
            i = len(job["translations"])
            job["translations"].extend(translate_batch(texts[i : i + BATCH_SIZE]))
            job["elapsed_secs"] += time.time() - step_start
        step_start = time.time()
        cefr_level, cefr_rationale = estimate_cefr(texts)
        job["elapsed_secs"] += time.time() - step_start
    except Exception as e:
        status_code = e.status_code if isinstance(e, HTTPException) else 500
        message = e.detail if isinstance(e, HTTPException) else f"Unexpected error: {e}"
        pending_id = save_pending(job, message, pending_id)
        raise HTTPException(status_code=status_code, detail={"message": message, "pending_id": pending_id}) from e

    output = [
        {
            "timestamp": f"{format_timestamp(seg['start'])} – {format_timestamp(seg['end'])}",
            "spanish": seg["text"],
            "english": job["translations"][i],
        }
        for i, seg in enumerate(job["segments"])
    ]
    session_id = save_session(job["filename"], output, cefr_level, cefr_rationale,
                              job["segments"][-1]["end"], job["elapsed_secs"], job["source_url"])
    if pending_id is not None:
        conn = get_db()
        conn.execute("DELETE FROM pending_transcriptions WHERE id = ?", (pending_id,))
        conn.commit()
        conn.close()
    return {
        "segments": output,
        "session_id": session_id,
        "cefr_level": cefr_level,
        "cefr_rationale": cefr_rationale,
        "filename": job["filename"],
        "source_url": job["source_url"],
    }


def save_pending(job: dict, error: str, pending_id: int | None = None) -> int:
    values = (job["filename"], job["source_url"], json.dumps(job["segments"]),
              json.dumps(job["translations"]), job["elapsed_secs"], error,
              datetime.now(timezone.utc).isoformat())
    conn = get_db()
    if pending_id is None:
        pending_id = conn.execute(
            """INSERT INTO pending_transcriptions
                   (filename, source_url, segments, translations, elapsed_secs, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            values,
        ).lastrowid
    else:
        conn.execute(
            """UPDATE pending_transcriptions SET filename = ?, source_url = ?, segments = ?,
                   translations = ?, elapsed_secs = ?, error = ?, created_at = ?
               WHERE id = ?""",
            (*values, pending_id),
        )
    conn.commit()
    conn.close()
    return pending_id


@router.get("/pending-transcriptions")
def list_pending():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, source_url, segments, translations, error, created_at "
        "FROM pending_transcriptions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return {"pending": [
        {
            "id": r["id"],
            "filename": r["filename"],
            "source_url": r["source_url"],
            "error": r["error"],
            "created_at": r["created_at"],
            "translated": len(json.loads(r["translations"])),
            "total": len(json.loads(r["segments"])),
        }
        for r in rows
    ]}


@router.post("/pending-transcriptions/{pending_id}/resume")
def resume_pending(pending_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM pending_transcriptions WHERE id = ?", (pending_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="That unfinished transcription no longer exists.")
    job = {
        "filename": row["filename"],
        "source_url": row["source_url"],
        "segments": json.loads(row["segments"]),
        "translations": json.loads(row["translations"]),
        "elapsed_secs": row["elapsed_secs"],
    }
    return finish_transcription(job, pending_id)


@router.delete("/pending-transcriptions/{pending_id}")
def delete_pending(pending_id: int):
    conn = get_db()
    conn.execute("DELETE FROM pending_transcriptions WHERE id = ?", (pending_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


# ── YouTube ──────────────────────────────────────────────────────────────────

# yt-dlp error text that means "you need to be logged in" (members-only,
# age-restricted, private, or YouTube's "confirm you're not a bot" check).
LOGIN_REQUIRED_RE = re.compile(r"sign in|log in|login|members|join this channel|cookies|private video", re.I)
COOKIES_UPLOADED_AT_KEY = "youtube_cookies_uploaded_at"
MAX_COOKIES_FILE_BYTES = 1_000_000


def normalize_youtube_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Paste a YouTube URL.")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise HTTPException(status_code=400, detail="Only YouTube URLs are supported (youtube.com or youtu.be).")
    return url


def ytdlp_options(**extra) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True}
    if YOUTUBE_COOKIES_PATH.exists():
        opts["cookiefile"] = str(YOUTUBE_COOKIES_PATH)
    return {**opts, **extra}


def youtube_error(e: Exception) -> HTTPException:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(getattr(e, "msg", None) or e))
    msg = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?", "", msg).strip()
    if LOGIN_REQUIRED_RE.search(msg):
        hint = ("Your uploaded cookies may have expired. Upload a fresh cookies.txt in Settings → YouTube."
                if YOUTUBE_COOKIES_PATH.exists() else
                "Upload a cookies.txt in Settings → YouTube (hover the ? there for instructions).")
        return HTTPException(status_code=400, detail=f"YouTube login required. {hint} (YouTube said: {msg})")
    return HTTPException(status_code=400, detail=f"Could not get the video from YouTube: {msg}")


def check_single_video(info: dict) -> None:
    if info.get("_type") == "playlist":
        raise HTTPException(status_code=400, detail="That's a playlist link. Paste the URL of a single video.")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise HTTPException(status_code=400, detail="Live streams and upcoming premieres can't be transcribed.")


def download_youtube_audio(url: str, dest_dir: Path) -> tuple[dict, Path]:
    # Plain bestaudio (usually webm/m4a): Whisper decodes it via ffmpeg, so
    # converting to mp3 like the download scripts do would only cost time.
    opts = ytdlp_options(format="bestaudio/best", outtmpl=str(dest_dir / "audio.%(ext)s"))
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            check_single_video(info)
            ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise youtube_error(e)
    files = [p for p in dest_dir.iterdir() if p.is_file()]
    if not files:
        raise HTTPException(status_code=502, detail="YouTube download finished but produced no audio file.")
    return info, files[0]


class YoutubeRequest(BaseModel):
    url: str


@router.get("/youtube-info")
def get_youtube_info(url: str):
    url = normalize_youtube_url(url)
    try:
        with yt_dlp.YoutubeDL(ytdlp_options()) as ydl:
            info = ydl.extract_info(url, download=False, process=False)
    except yt_dlp.utils.DownloadError as e:
        raise youtube_error(e)
    check_single_video(info)
    return {
        "title": info.get("title") or url,
        "duration": info.get("duration"),
        "channel": info.get("channel") or info.get("uploader") or "",
    }


@router.post("/transcribe-url")
def transcribe_url(req: YoutubeRequest):
    url = normalize_youtube_url(req.url)
    with tempfile.TemporaryDirectory() as tmp:
        info, audio_path = download_youtube_audio(url, Path(tmp))
        return process_audio(str(audio_path), info.get("title") or url, info.get("webpage_url") or url)


def youtube_cookies_status() -> dict:
    conn = get_db()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (COOKIES_UPLOADED_AT_KEY,)).fetchone()
    conn.close()
    uploaded = YOUTUBE_COOKIES_PATH.exists()
    return {"uploaded": uploaded, "uploaded_at": row["value"] if uploaded and row else None}


@router.get("/youtube-cookies")
def read_youtube_cookies():
    return youtube_cookies_status()


@router.put("/youtube-cookies")
async def upload_youtube_cookies(file: UploadFile = File(...)):
    data = await file.read(MAX_COOKIES_FILE_BYTES + 1)
    if len(data) > MAX_COOKIES_FILE_BYTES:
        raise HTTPException(status_code=400, detail="That file is too large to be a cookies.txt export.")
    # Windows exports use CRLF, which yt-dlp on Linux mishandles.
    data = data.replace(b"\r\n", b"\n")

    # Validate with yt-dlp's own loader before replacing the current file.
    tmp_path = YOUTUBE_COOKIES_PATH.with_suffix(".upload")
    tmp_path.write_bytes(data)
    os.chmod(tmp_path, 0o600)
    try:
        jar = yt_dlp.cookies.YoutubeDLCookieJar(str(tmp_path))
        jar.load()
        youtube_count = sum(1 for c in jar if c.domain.lstrip(".").endswith("youtube.com"))
    except Exception as e:
        print(f"cookies upload rejected: {e!r}")
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Not a valid cookies.txt file. It must be in Netscape cookies.txt format (not JSON), "
                   "starting with a '# Netscape HTTP Cookie File' line.",
        )
    if youtube_count == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="That file has no youtube.com cookies. Export it while on a youtube.com tab.")
    os.replace(tmp_path, YOUTUBE_COOKIES_PATH)

    conn = get_db()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (COOKIES_UPLOADED_AT_KEY, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return youtube_cookies_status()


@router.delete("/youtube-cookies")
def delete_youtube_cookies():
    YOUTUBE_COOKIES_PATH.unlink(missing_ok=True)
    conn = get_db()
    conn.execute("DELETE FROM settings WHERE key = ?", (COOKIES_UPLOADED_AT_KEY,))
    conn.commit()
    conn.close()
    return youtube_cookies_status()


@router.get("/sessions")
def list_sessions():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, transcribed_at, cefr_level, source_url FROM sessions ORDER BY transcribed_at DESC"
    ).fetchall()
    conn.close()
    return {"sessions": [dict(r) for r in rows]}


@router.get("/sessions/{session_id}")
def get_session(session_id: int):
    conn = get_db()
    session = conn.execute(
        "SELECT id, filename, transcribed_at, cefr_level, cefr_rationale, source_url FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    segs = conn.execute(
        "SELECT timestamp, spanish, english FROM segments WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return {**dict(session), "segments": [dict(s) for s in segs]}


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Please upload an audio file (mp3, wav, m4a, ogg, flac).")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return process_audio(tmp_path, file.filename)
    finally:
        os.unlink(tmp_path)


@router.get("/transcription-estimate")
def get_transcription_estimate(duration: float):
    conn = get_db()
    rows = conn.execute(
        """SELECT audio_duration_secs, transcription_secs FROM sessions
           WHERE audio_duration_secs > 0 AND transcription_secs > 0"""
    ).fetchall()
    conn.close()
    if not rows:
        return {"estimated_secs": None, "based_on": 0}
    ratios = [r["transcription_secs"] / r["audio_duration_secs"] for r in rows]
    avg_ratio = sum(ratios) / len(ratios)
    return {"estimated_secs": round(duration * avg_ratio), "based_on": len(rows)}


@router.get("/sessions/{session_id}/quiz")
def get_quiz(session_id: int):
    conn = get_db()
    cached = conn.execute(
        "SELECT questions FROM quiz_cache WHERE session_id = ?", (session_id,)
    ).fetchone()
    if cached:
        conn.close()
        return {"questions": json.loads(cached["questions"])}

    segs = conn.execute(
        "SELECT spanish FROM segments WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()
    conn.close()
    if not segs:
        raise HTTPException(status_code=404, detail="No segments found")

    text = " ".join(r["spanish"] for r in segs)[:4000]
    raw = strip_code_fences(complete(
        "Based on this Spanish transcript, write 5 comprehension questions in Spanish with answers in Spanish "
        "and an English translation of each answer. "
        "Return JSON only (no markdown):\n"
        '[{"question": "¿...?", "answer_es": "...", "answer_en": "..."}, ...]\n\n'
        + text
    ))
    try:
        questions = json.loads(raw)
    except Exception as e:
        print(f"quiz parse error: {e!r} — raw: {raw!r}")
        raise HTTPException(status_code=500, detail="Failed to parse quiz response")

    conn = get_db()
    conn.execute(
        """INSERT INTO quiz_cache (session_id, questions, cached_at) VALUES (?, ?, ?)
           ON CONFLICT(session_id) DO NOTHING""",
        (session_id, json.dumps(questions), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return {"questions": questions}


@router.get("/search")
def search_segments(q: str = ""):
    q = q.strip()
    if not q:
        return {"results": [], "query": q}
    pattern = f"%{q}%"
    conn = get_db()
    rows = conn.execute(
        """SELECT seg.session_id, seg.timestamp, seg.spanish, seg.english,
                  s.filename, s.transcribed_at
           FROM segments seg
           JOIN sessions s ON seg.session_id = s.id
           WHERE LOWER(seg.spanish) LIKE LOWER(?) OR LOWER(seg.english) LIKE LOWER(?)
           ORDER BY s.transcribed_at DESC, seg.id
           LIMIT 200""",
        (pattern, pattern),
    ).fetchall()
    conn.close()
    return {"results": [dict(r) for r in rows], "query": q}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int):
    conn = get_db()
    if not conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    conn.execute("DELETE FROM segments WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


class LookupRequest(BaseModel):
    word: str
    context: str = ""


class WordSave(BaseModel):
    word: str
    part_of_speech: str = ""
    definition: str = ""
    example_es: str = ""
    example_en: str = ""


@router.post("/lookup")
def lookup_word(req: LookupRequest):
    word = req.word.lower()
    conn = get_db()
    cached = conn.execute(
        "SELECT part_of_speech, definition, example_es, example_en FROM word_cache WHERE word = ?",
        (word,),
    ).fetchone()
    conn.close()
    if cached:
        return dict(cached)

    raw = strip_code_fences(complete(
        f'Define the Spanish word "{word}" as used in this sentence: "{req.context}"\n\n'
        'Reply with JSON only (no markdown):\n'
        '{"part_of_speech": "noun", "definition": "English definition", '
        '"example_es": "Spanish example sentence", "example_en": "English translation of example"}'
    ))
    try:
        result = json.loads(raw)
    except Exception as e:
        print(f"lookup_word parse error: {e!r} — raw: {raw!r}")
        raise HTTPException(status_code=500, detail="Failed to parse lookup response")

    conn = get_db()
    conn.execute(
        """INSERT INTO word_cache (word, part_of_speech, definition, example_es, example_en, cached_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(word) DO NOTHING""",
        (word, result.get("part_of_speech", ""), result.get("definition", ""),
         result.get("example_es", ""), result.get("example_en", ""),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return result


@router.get("/word-list")
def get_word_list():
    conn = get_db()
    rows = conn.execute(
        "SELECT word, part_of_speech, definition, example_es, example_en FROM word_list ORDER BY added_at DESC"
    ).fetchall()
    conn.close()
    return {"words": [dict(r) for r in rows]}


@router.post("/word-list")
def save_word(entry: WordSave):
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO word_list (word, part_of_speech, definition, example_es, example_en, added_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(word) DO UPDATE SET
             part_of_speech=excluded.part_of_speech,
             definition=excluded.definition,
             example_es=excluded.example_es,
             example_en=excluded.example_en""",
        (entry.word, entry.part_of_speech, entry.definition, entry.example_es, entry.example_en, now),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@router.delete("/word-list/{word}")
def delete_word(word: str):
    conn = get_db()
    conn.execute("DELETE FROM word_list WHERE word = ?", (word,))
    conn.commit()
    conn.close()
    return {"ok": True}


class SettingsUpdate(BaseModel):
    provider: str
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = ""


@router.get("/settings")
def read_settings():
    return get_settings()


def validate_settings(update: SettingsUpdate) -> dict:
    if update.provider not in ("anthropic", "ollama"):
        raise HTTPException(status_code=400, detail="provider must be 'anthropic' or 'ollama'")
    if update.provider == "anthropic" and not update.anthropic_model.strip():
        raise HTTPException(status_code=400, detail="Choose an Anthropic model.")
    if update.provider == "ollama" and not update.ollama_model.strip():
        raise HTTPException(status_code=400, detail="Choose an Ollama model.")
    return {
        "provider": update.provider,
        "anthropic_model": update.anthropic_model.strip(),
        "ollama_host": normalize_ollama_host(update.ollama_host or DEFAULT_OLLAMA_HOST),
        "ollama_model": update.ollama_model.strip(),
    }


@router.put("/settings")
def update_settings(update: SettingsUpdate):
    values = validate_settings(update)
    conn = get_db()
    conn.executemany(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        list(values.items()),
    )
    conn.commit()
    conn.close()
    return get_settings()


@router.post("/settings/test")
def test_settings(update: SettingsUpdate):
    """Send a tiny prompt using the given (possibly unsaved) settings."""
    settings = validate_settings(update)
    start = time.time()
    reply = complete("Reply with just the word OK.", settings, ollama_timeout=OLLAMA_TEST_TIMEOUT_SECS)
    if not reply:
        raise HTTPException(status_code=502, detail="Connected, but the model returned an empty reply.")
    return {"secs": round(time.time() - start, 1), "reply": reply[:200]}


@router.get("/models/anthropic")
def list_anthropic_models():
    try:
        models = [{"id": m.id, "display_name": m.display_name} for m in get_claude().models.list()]
        return {"models": models, "error": None}
    except Exception as e:
        print(f"list_anthropic_models failed: {e!r}")
        return {"models": FALLBACK_ANTHROPIC_MODELS, "error": "Could not fetch models from Anthropic; showing a default list."}


@router.get("/models/ollama")
def list_ollama_models(host: str):
    result = ollama_request(host, "/api/tags")
    names = sorted(m.get("name", "") for m in result.get("models", []) if m.get("name"))
    return {"models": names}


@router.get("/config.js")
def config_js():
    return Response(content=f"window.BASE_PATH = {json.dumps(BASE_PATH)};", media_type="application/javascript")


app.include_router(router)
app.mount(f"{BASE_PATH}/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
