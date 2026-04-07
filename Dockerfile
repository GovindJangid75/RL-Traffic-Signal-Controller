# ─────────────────────────────────────────────────────────────────────────────
# Traffic RL v2 — Multi-stage Dockerfile
# Stages: base → app | api
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="Traffic RL Team"
LABEL version="2.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN touch env/__init__.py agent/__init__.py api/__init__.py

# ── Streamlit target ──────────────────────────────────────────────────────────
FROM base AS app

RUN mkdir -p /root/.streamlit && \
    printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml && \
    printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\n\n[browser]\ngatherUsageStats = false\n' \
           > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]

# ── FastAPI target ────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "api.server:app", \
            "--host", "0.0.0.0", "--port", "8000"]