# AI Every Time, ready to run anywhere that runs containers. The database lives in /data:
# mount a volume there so learners' accounts survive restarts and new versions.
# Build from this folder:  docker build -t ai-every-time .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AIET_DATABASE=/data/ai_every_time.db

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY static ./static

# Run as an ordinary user that owns only the data folder.
RUN useradd --create-home --uid 10001 aiet && mkdir -p /data && chown aiet /data
USER aiet
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]

# One process: the loops that send reminder emails and notifications and write scheduled posts run
# inside it, so run one container per database (see "Deploy it" in the README).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
