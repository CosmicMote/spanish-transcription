import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DB_PATH = Path(__file__).parent.parent / "transcriptions.db"
BATCH_SIZE = 25

app = FastAPI()

print(f"Loading Whisper model '{WHISPER_MODEL}'...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded.")

claude = anthropic.Anthropic()


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
    """)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "cefr_level" not in existing:
        conn.execute("ALTER TABLE sessions ADD COLUMN cefr_level TEXT")
    if "cefr_rationale" not in existing:
        conn.execute("ALTER TABLE sessions ADD COLUMN cefr_rationale TEXT")
    conn.commit()
    conn.close()


init_db()


def save_session(filename: str, segments: list[dict], cefr_level: str = "", cefr_rationale: str = "") -> int:
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO sessions (filename, transcribed_at, cefr_level, cefr_rationale) VALUES (?, ?, ?, ?)",
        (filename, now, cefr_level, cefr_rationale),
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
    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": (
                    "Translate each numbered Spanish phrase to English. "
                    "Return only the numbered translations in the same format "
                    '(e.g., "1. The translation"). Do not add explanations.\n\n'
                    + numbered
                ),
            }
        ],
    )
    response_text = message.content[0].text.strip()
    translations = {}
    for line in response_text.splitlines():
        match = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if match:
            idx = int(match.group(1)) - 1
            translations[idx] = match.group(2)
    return [translations.get(i, "") for i in range(len(spanish_texts))]


def estimate_cefr(spanish_texts: list[str]) -> tuple[str, str]:
    sample = "\n".join(spanish_texts[:40])[:3000]
    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    "Estimate the CEFR difficulty level (A1, A2, B1, B2, C1, or C2) "
                    "of the following Spanish transcript for a language learner. "
                    "Consider vocabulary range, sentence complexity, and topic familiarity. "
                    'Reply with JSON only: {"level": "B1", "rationale": "one sentence"}. '
                    "No markdown, no extra text.\n\n" + sample
                ),
            }
        ],
    )
    raw = message.content[0].text.strip()
    # strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
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


@app.get("/sessions")
def list_sessions():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, transcribed_at, cefr_level FROM sessions ORDER BY transcribed_at DESC"
    ).fetchall()
    conn.close()
    return {"sessions": [dict(r) for r in rows]}


@app.get("/sessions/{session_id}")
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


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Please upload an audio file (mp3, wav, m4a, ogg, flac).")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path, language="es", task="transcribe")
        segments = result["segments"]

        if not segments:
            return {"segments": [], "session_id": None}

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

        session_id = save_session(file.filename, output, cefr_level, cefr_rationale)
        return {"segments": output, "session_id": session_id, "cefr_level": cefr_level, "cefr_rationale": cefr_rationale}
    finally:
        os.unlink(tmp_path)


@app.get("/search")
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


@app.delete("/sessions/{session_id}")
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


@app.post("/lookup")
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

    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f'Define the Spanish word "{word}" as used in this sentence: "{req.context}"\n\n'
                'Reply with JSON only (no markdown):\n'
                '{"part_of_speech": "noun", "definition": "English definition", '
                '"example_es": "Spanish example sentence", "example_en": "English translation of example"}'
            ),
        }],
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
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


@app.get("/word-list")
def get_word_list():
    conn = get_db()
    rows = conn.execute(
        "SELECT word, part_of_speech, definition, example_es, example_en FROM word_list ORDER BY added_at DESC"
    ).fetchall()
    conn.close()
    return {"words": [dict(r) for r in rows]}


@app.post("/word-list")
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


@app.delete("/word-list/{word}")
def delete_word(word: str):
    conn = get_db()
    conn.execute("DELETE FROM word_list WHERE word = ?", (word,))
    conn.commit()
    conn.close()
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
