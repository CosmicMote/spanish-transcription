FROM python:3.12-slim

# ffmpeg is required by Whisper for audio file decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer — only reruns if requirements.txt changes)
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper model so the image is self-contained.
# Override at build time with: docker build --build-arg WHISPER_MODEL=medium .
# The ENV below ensures the app uses the same model at runtime.
ARG WHISPER_MODEL=small
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN python -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# The SQLite database is written to /data so it can be persisted with a volume:
#   docker run -v spanish_data:/data ...
ENV DB_PATH=/data/transcriptions.db
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
