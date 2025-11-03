# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install ffmpeg (required by pydub)
RUN apt-get update && apt-get install -y ffmpeg build-essential libsndfile1 --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create output & tmp dirs
RUN mkdir -p /app/podcasts /tmp/podcast_segments
ENV OUTPUT_DIR=/app/podcasts
ENV TMP_ROOT=/tmp/podcast_segments

# Use uvicorn to serve.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
