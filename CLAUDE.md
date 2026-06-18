# Spanish Transcription

This is an application which allows a user to upload an mp3 file of
spanish audio. The application transcribes the audio into spanish
text, translates it into English, and has a UI to display both
side by side. Other features include showing definitions of spanish
words and saving past sessions.

The tech stack on the BE is FastAPI using OpenAI's Whisper (local model) for transcription
and the claude APIs for translation. The tech stack on the FE is very minimal,
just plain old HTML+JavaScript+CSS.

## Commands
- Set up virtual environment: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Configuration: Set `ANTHROPIC_API_KEY` environment variable with Anthropic API key for translation
- Configuration: Set `WHISPER_MODEL` to tiny, base, small, medium, large (optional, defaults to small)
- Configuration: Set `DB_PATH` for location of SQLite DB file (optional, can generally be left to default)
- Run locally: `uvicorn backend.main:app --reload --port 8000`
- Nothing needed to run frontend since it's just served by the FastAPI server

## Docker
- Build: `docker build -t spanish-transcription .`
- Build with non-default whisper model: `docker build --build-arg WHISPER_MODEL=medium -t spanish-transcription .`
- Run: `docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_api_key_here -v spanish_data:/data spanish-transcription`
- Access UI after running: `http://localhost:8000`

## Testing
No tests currently.

## Structure
- `backend/` — Backend Python code
- `frontend/` — Frontend HTML+CSS+JavaScript code

## Architecture notes
- Backend is a simple FastAPI server using a SQLite database for storing past sessions.
- Use OpenAI local whisper model for transcription to reduce cost
- Frontend is a single file containing all HTML+CSS+JavaScript
- Key architectural driver is simplicity at this point

## Constraints
- When installing a Python library, always do this in the virtual environment, never in the global Python environment
- Prefer simplicity over abstraction; no additional dependencies without discussion