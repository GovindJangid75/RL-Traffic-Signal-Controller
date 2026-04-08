# ─────────────────────────────────────────────────────────────────────────────
# Traffic RL v2 — Multi-stage Dockerfile (HF Compatible)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

LABEL maintainer="Traffic RL Team"
LABEL version="2.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN touch env/__init__.py agent/__init__.py api/__init__.py

# ── Streamlit App (MAIN for HF) ───────────────────────────────────────────────
FROM base AS app

RUN mkdir -p /root/.streamlit && \
    printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml && \
    printf '[server]\nheadless = true\nport = 7860\nenableCORS = false\n\n[browser]\ngatherUsageStats = false\n' \
           > /root/.streamlit/config.toml

# IMPORTANT: HF uses port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=7860", "--server.address=0.0.0.0"]

# ── FastAPI (optional, not used by HF UI) ─────────────────────────────────────
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "api.server:app", \
            "--host", "0.0.0.0", "--port", "8000"]