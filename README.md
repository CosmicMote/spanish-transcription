# Spanish Transcription

A web app for transcribing Spanish audio files into timestamped bilingual transcripts, with vocabulary tools and comprehension features for language learners.

Audio is transcribed locally using [OpenAI Whisper](https://github.com/openai/whisper) (no transcription API costs), and English translations are generated via the Anthropic API (Claude Haiku).

## Features

- **Bilingual transcripts** — Timestamped table of Spanish segments alongside English translations
- **CEFR difficulty rating** — Each episode is automatically rated A1–C2 with a rationale; hover the badge for a description of each level
- **Vocabulary lookup** — Click any Spanish word in the transcript to see its definition, part of speech, and an example sentence; lookups are cached in the database so each word is only sent to the API once
- **Saved word list** — Save looked-up words to a persistent personal vocabulary list
- **Comprehension questions** — Generate 5 Spanish comprehension questions for any episode with a "Quiz me" button; questions are cached per session
- **Full-text search** — Search across all saved transcripts in both Spanish and English
- **Past sessions** — Browse and reload any previously transcribed episode; delete sessions you no longer need
- **Transcription time estimates** — After the first session, the app estimates how long a new transcription will take based on the audio duration and past timing data

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Transcription | OpenAI Whisper (runs locally) |
| Translation / AI | Anthropic Claude Haiku |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Database | SQLite |

---

## Running in development

### Prerequisites

- Python 3.12
- `ffmpeg` installed on your system (required by Whisper for audio decoding)
- An Anthropic API key

### Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd spanish-transcription
   ```

2. **Create a `.env` file** in the project root:

   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

3. **Create a virtual environment and install dependencies**

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Start the server**

   ```bash
   cd ..
   uvicorn backend.main:app --reload --port 8000
   ```

5. Open [http://localhost:8000](http://localhost:8000) in your browser.

### Whisper model

The app uses the `small` Whisper model by default. You can override this with the `WHISPER_MODEL` environment variable:

```bash
WHISPER_MODEL=medium uvicorn backend.main:app --reload --port 8000
```

Available models (larger = more accurate but slower): `tiny`, `base`, `small`, `medium`, `large`.

---

## Running with Docker

### Prerequisites

- Docker installed
- An Anthropic API key

### Build the image

```bash
docker build -t spanish-transcription .
```

This bakes the `small` Whisper model into the image. To use a different model:

```bash
docker build --build-arg WHISPER_MODEL=medium -t spanish-transcription .
```

> **Note:** The first build will take several minutes — PyTorch and the Whisper model are large downloads (~500 MB for the default `small` build).

### Run the container

```bash
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_api_key_here \
  -v spanish_data:/data \
  spanish-transcription
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

The `-v spanish_data:/data` flag mounts a Docker volume so that your transcription database (sessions, word list, caches) persists across container restarts. Without it, all data is lost when the container stops.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `WHISPER_MODEL` | `small` | Whisper model to use at runtime (should match the model baked in at build time) |
| `DB_PATH` | `/data/transcriptions.db` | Path to the SQLite database file |
