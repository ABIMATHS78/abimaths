# AI Every Time, ready to run anywhere that runs containers. The database lives in /data:
# mount a volume or disk there so learners' accounts survive restarts and new versions.
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
# For bringing the old site's members over from the command line (the Staff page's Setup tab does it too).
COPY scripts/import_legacy.py ./scripts/import_legacy.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# The app runs as an ordinary user that owns only the data folder. The entrypoint starts as root just
# long enough to hand it a disk the host attached as root's. The .ssh folders let hosts that offer SSH
# into a container (Render, for one) connect.
RUN useradd --create-home --uid 10001 aiet && mkdir -p /data && chown aiet:aiet /data \
    && mkdir -p /root/.ssh /home/aiet/.ssh && chmod 0700 /root/.ssh /home/aiet/.ssh && chown aiet:aiet /home/aiet/.ssh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh
EXPOSE 8000

# Straight to the app, never through a proxy that the host's settings might name.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8000/healthz', timeout=4)"]

# One process (whatever WEB_CONCURRENCY says): the loops that send reminder emails and notifications
# and write scheduled posts run inside it, so run one container per database (see "Deploy it").
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
