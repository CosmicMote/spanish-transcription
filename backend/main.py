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
    """)
    conn.commit()
    conn.close()


init_db()


def save_session(filename: str, segments: list[dict]) -> int:
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO sessions (filename, transcribed_at) VALUES (?, ?)",
        (filename, now),
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
        "SELECT id, filename, transcribed_at FROM sessions ORDER BY transcribed_at DESC"
    ).fetchall()
    conn.close()
    return {"sessions": [dict(r) for r in rows]}


@app.get("/sessions/{session_id}")
def get_session(session_id: int):
    conn = get_db()
    session = conn.execute(
        "SELECT id, filename, transcribed_at FROM sessions WHERE id = ?", (session_id,)
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

        session_id = save_session(file.filename, output)
        return {"segments": output, "session_id": session_id}
    finally:
        os.unlink(tmp_path)


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
