import json
import os
import re
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import whisper
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DB_PATH = Path(os.getenv("DB_PATH", str(Path(__file__).parent.parent / "transcriptions.db")))
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


def complete(prompt: str) -> str:
    """Send a single-turn prompt to the configured AI provider and return its text reply."""
    settings = get_settings()
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
            timeout=OLLAMA_TIMEOUT_SECS,
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
    # Skip thinking blocks; only text blocks hold the answer.
    return "".join(b.text for b in message.content if b.type == "text").strip()


def strip_code_fences(raw: str) -> str:
    return re.sub(r"^```[a-z]*\n?", "", raw.strip()).rstrip("`").strip()


def save_session(filename: str, segments: list[dict], cefr_level: str = "", cefr_rationale: str = "",
                 audio_duration_secs: float | None = None, transcription_secs: float | None = None) -> int:
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO sessions (filename, transcribed_at, cefr_level, cefr_rationale, audio_duration_secs, transcription_secs) VALUES (?, ?, ?, ?, ?, ?)",
        (filename, now, cefr_level, cefr_rationale, audio_duration_secs, transcription_secs),
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


def translate_segments(segments: list[dict]) -> list[str]:
    texts = [seg["text"].strip() for seg in segments]
    all_translations: list[str] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        all_translations.extend(translate_batch(batch))
    return all_translations


@router.get("/sessions")
def list_sessions():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, transcribed_at, cefr_level FROM sessions ORDER BY transcribed_at DESC"
    ).fetchall()
    conn.close()
    return {"sessions": [dict(r) for r in rows]}


@router.get("/sessions/{session_id}")
def get_session(session_id: int):
    conn = get_db()
    session = conn.execute(
        "SELECT id, filename, transcribed_at, cefr_level, cefr_rationale FROM sessions WHERE id = ?", (session_id,)
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
        processing_start = time.time()
        result = whisper_model.transcribe(tmp_path, language="es", task="transcribe")
        segments = result["segments"]

        if not segments:
            return {"segments": [], "session_id": None}

        audio_duration_secs = segments[-1]["end"] if segments else None

        translations = translate_segments(segments)

        output = [
            {
                "timestamp": f"{format_timestamp(seg['start'])} – {format_timestamp(seg['end'])}",
                "spanish": seg["text"].strip(),
                "english": translations[i],
            }
            for i, seg in enumerate(segments)
        ]

        spanish_texts = [s["spanish"] for s in output]
        cefr_level, cefr_rationale = estimate_cefr(spanish_texts)
        transcription_secs = time.time() - processing_start

        session_id = save_session(file.filename, output, cefr_level, cefr_rationale, audio_duration_secs, transcription_secs)
        return {"segments": output, "session_id": session_id, "cefr_level": cefr_level, "cefr_rationale": cefr_rationale}
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


@router.put("/settings")
def update_settings(update: SettingsUpdate):
    if update.provider not in ("anthropic", "ollama"):
        raise HTTPException(status_code=400, detail="provider must be 'anthropic' or 'ollama'")
    if update.provider == "anthropic" and not update.anthropic_model.strip():
        raise HTTPException(status_code=400, detail="Choose an Anthropic model.")
    if update.provider == "ollama" and not update.ollama_model.strip():
        raise HTTPException(status_code=400, detail="Choose an Ollama model.")
    values = {
        "provider": update.provider,
        "anthropic_model": update.anthropic_model.strip(),
        "ollama_host": normalize_ollama_host(update.ollama_host or DEFAULT_OLLAMA_HOST),
        "ollama_model": update.ollama_model.strip(),
    }
    conn = get_db()
    conn.executemany(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        list(values.items()),
    )
    conn.commit()
    conn.close()
    return get_settings()


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
