# Spanish Transcription

A web app for transcribing Spanish audio files into timestamped bilingual transcripts, with vocabulary tools and comprehension features for language learners.

Audio is transcribed locally using [OpenAI Whisper](https://github.com/openai/whisper) (no transcription API costs), and English translations and other AI features use either the Anthropic API (Claude Haiku by default) or a local model served by [Ollama](https://ollama.com), selectable in the app.

## Features

- **Bilingual transcripts** — Timestamped table of Spanish segments alongside English translations
- **YouTube links** — Paste a YouTube URL instead of uploading a file, including members-only videos once you upload your login cookies (see [Transcribing YouTube videos](#transcribing-youtube-videos))
- **CEFR difficulty rating** — Each episode is automatically rated A1–C2 with a rationale; hover the badge for a description of each level
- **Vocabulary lookup** — Click any Spanish word in the transcript to see its definition, part of speech, and an example sentence; lookups are cached in the database so each word is only sent to the API once
- **Saved word list** — Save looked-up words to a persistent personal vocabulary list
- **Comprehension questions** — Generate 5 Spanish comprehension questions for any episode with a "Quiz me" button; questions are cached per session
- **Full-text search** — Search across all saved transcripts in both Spanish and English
- **Past sessions** — Browse and reload any previously transcribed episode; delete sessions you no longer need
- **Configurable AI model** — Click the ⚙ button in the header to pick an Anthropic model, or point the app at an Ollama host and pick any model it has installed; the choice is saved in the database
- **Transcription time estimates** — After the first session, the app estimates how long a new transcription will take based on the audio duration and past timing data

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Transcription | OpenAI Whisper (runs locally) |
| Translation / AI | Anthropic Claude (any model) or Ollama (local models) |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Database | SQLite |

---

## Running in development

### Prerequisites

- Python 3.12
- `ffmpeg` installed on your system (required by Whisper for audio decoding)
- An Anthropic API key, or an [Ollama](https://ollama.com) server with at least one model pulled

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

### AI model

By default the app uses Claude Haiku via the Anthropic API. To change it, click the ⚙ button in the header:

- **Anthropic** — choose any model available to your API key.
- **Ollama (local)** — enter the Ollama host URL (e.g. `http://localhost:11434`), click **Load models**, and choose one of the models installed on that host.

Local models are slower and their JSON output can be less reliable than Claude's, so word lookups, CEFR ratings, and quizzes may occasionally fail with smaller models.

### Running behind a reverse proxy under a URL path

By default the app runs at the root path (`/`). To host it under a path prefix behind a reverse proxy (e.g. `https://example.com/spanish-transcription`), set the `BASE_PATH` environment variable and configure your reverse proxy to forward the full path unstripped (i.e. requests to `/spanish-transcription/...` should reach the app as `/spanish-transcription/...`, not have the prefix stripped):

```bash
BASE_PATH=/spanish-transcription uvicorn backend.main:app --reload --port 8000
```

Then access the app at `http://localhost:8000/spanish-transcription`.

---

## Running with Docker

### Prerequisites

- Docker installed
- An Anthropic API key, or an [Ollama](https://ollama.com) server with at least one model pulled

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
docker run -d --restart unless-stopped -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_api_key_here \
  -v spanish_data:/data \
  spanish-transcription
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

The `-v spanish_data:/data` flag mounts a Docker volume so that your transcription database (sessions, word list, caches) persists across container restarts. Without it, all data is lost when the container stops.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(none)* | Your Anthropic API key (required when using Anthropic models) |
| `OLLAMA_HOST` | `http://localhost:11434` | Initial Ollama host shown in the AI settings dialog. From inside Docker, use `http://host.docker.internal:11434` to reach Ollama on the host machine |
| `WHISPER_MODEL` | `small` | Whisper model to use at runtime (should match the model baked in at build time) |
| `DB_PATH` | `/data/transcriptions.db` | Path to the SQLite database file |
| `BASE_PATH` | *(none)* | URL path prefix to run the app under, e.g. `/spanish-transcription` (see [above](#running-behind-a-reverse-proxy-under-a-url-path)) |

---

## Transcribing YouTube videos

Choose **YouTube URL** above the Transcribe button and paste a video link (`youtube.com/watch?v=…` or `youtu.be/…`). The app shows the video's title and length, plus an estimated transcription time. When you click Transcribe, the server downloads the audio with [yt-dlp](https://github.com/yt-dlp/yt-dlp) and transcribes it. The session is named after the video, and **▶ YouTube** links in Past Sessions and the transcript header take you back to it.

Only single videos are supported; playlists and live streams are rejected. yt-dlp, the YouTube challenge-solver scripts, and the Deno JavaScript runtime it needs are all installed from `backend/requirements.txt`, so no extra system packages are required beyond ffmpeg. YouTube changes often. If downloads start failing for every video, update yt-dlp (`pip install -U "yt-dlp[default,deno]"`, or rebuild the Docker image).

### Members-only videos (cookies)

Public videos work as-is. Members-only, age-restricted, and other login-required videos need your YouTube login cookies. These are uploaded once in the app:

1. Click ⚙ in the header, then go to **YouTube → Login cookies**. Hover the **?** for step-by-step export instructions.
2. Following those steps, export a `cookies.txt` from a **private/incognito window** with the [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/) extension (Firefox) or [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) (Chrome). Then close the window. YouTube rotates cookies in open browser tabs, so cookies exported this way last much longer. The procedure comes from the [yt-dlp wiki](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies).
3. Click **Upload cookies.txt**.

The file is validated, then stored as `youtube_cookies.txt` next to the database (`/data` in Docker). It's used for every YouTube download. If a video fails with a "YouTube login required" error, the cookies have expired: export and upload a fresh file. Because the file lives on the server, this works even when the app runs on a different machine from your browser.

### Finding the YouTube URL

If the embedded player exposes a direct link to YouTube, just copy that. If it doesn't, you can still recover the URL:

1. Open the page with the embedded video and open your browser's dev tools (F12, or Cmd+Option+I on macOS) to the **Network** tab.
2. Reload the page and let the video start loading.
3. Filter for `embed` and look for a request like `GET https://www.youtube.com/embed/VIDEO_ID?...`.
4. Take the `VIDEO_ID` and open `https://www.youtube.com/watch?v=VIDEO_ID` in a new tab to confirm it's the right video.

---

## Downloading audio with the scripts (alternative)

If you'd rather download the mp3 yourself (for example, to keep a copy), the included scripts still work. Upload the result as you would any audio file:

- `download_spanish.sh` — macOS (and Linux)
- `download_spanish.ps1` — Windows (PowerShell)

### Prerequisites (one-time setup)

1. **Install yt-dlp and ffmpeg** (yt-dlp uses ffmpeg to convert the audio to mp3)

   macOS ([Homebrew](https://brew.sh)):

   ```bash
   brew install yt-dlp ffmpeg
   ```

   Windows:

   ```powershell
   winget install yt-dlp.yt-dlp
   ```

   Or download the standalone binary from the [yt-dlp releases page](https://github.com/yt-dlp/yt-dlp/releases).

2. **Log into YouTube** in Firefox with the account that has access to the content (e.g. a membership). Firefox works reliably on all platforms. On Windows, Chrome's cookie store has security protections that prevent yt-dlp from reading it. On macOS, Chrome generally works but will prompt for Keychain access the first time, and Safari requires granting your terminal Full Disk Access.

Once you have the URL ([how to find it](#finding-the-youtube-url)), run the script for your platform. The mp3 is saved to your Downloads folder by default; then upload it to the app as you would any audio file.

### Usage on macOS

```bash
./download_spanish.sh "https://www.youtube.com/watch?v=..."
```

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `-b` | `firefox` | Browser to read cookies from (`firefox`, `chrome`, `safari`, ...) |
| `-o` | `~/Downloads` | Folder to save the mp3 |

Example with overrides:

```bash
./download_spanish.sh -b chrome -o ~/Desktop "https://www.youtube.com/watch?v=..."
```

### Usage on Windows

```powershell
.\download_spanish.ps1 -Url "https://www.youtube.com/watch?v=..."
```

**Optional parameters:**

| Parameter | Default | Description |
|---|---|---|
| `-Url` | *(required)* | YouTube URL of the video |
| `-Browser` | `firefox` | Browser to read cookies from (`firefox`, `chrome`, `edge`) |
| `-OutputDir` | `%USERPROFILE%\Downloads` | Folder to save the mp3 |

Example with overrides:

```powershell
.\download_spanish.ps1 -Url "https://www.youtube.com/watch?v=..." -Browser firefox -OutputDir "C:\Users\jeff\Desktop"
```
