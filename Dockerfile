# The base image intentionally runs the transparent, precomputed demo without
# installing a large AMRFinderPlus database. See README.md for the production
# image extension and the version-pinning requirements for live annotation.
FROM python:3.12-slim-bookworm

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GENOME_FIREWALL_ENABLE_OPENAI=false \
    GENOME_FIREWALL_ALLOW_LIVE_ANNOTATION=false

WORKDIR /app

RUN groupadd --gid "${APP_GID}" appgroup \
    && useradd --uid "${APP_UID}" --gid appgroup --create-home --shell /usr/sbin/nologin appuser

COPY pyproject.toml requirements.txt README.md ./
COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps . \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
