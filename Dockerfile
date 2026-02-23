# ══════════════════════════════════════════════════════════════════════════════
# PhishTrace — Phishing Detection via Interaction Trace Graphs
# Multi-stage Docker build: Python + Playwright + TeX Live + Z3
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build-time proxy (pass via --build-arg or docker compose)
ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy="localhost,127.0.0.1"

# ─── System dependencies ─────────────────────────────────────────────────────
# NOTE: If BuildKit network fails, build manually:
#   docker run -d --name phishtrace-build python:3.11-slim sleep 3600
#   docker exec phishtrace-build apt-get update && apt-get install -y ...
#   docker exec phishtrace-build pip install ...
#   docker commit phishtrace-build phishtrace:latest
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    apt-get update && apt-get install -y --no-install-recommends \
    # TeX Live (minimal for LNCS/ICCS compilation)
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-bibtex-extra \
    texlive-science \
    # General utilities
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ─── Python dependencies ─────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir z3-solver && \
    pip install --no-cache-dir playwright && \
    playwright install chromium && \
    playwright install-deps chromium

# ─── Application code ────────────────────────────────────────────────────────
COPY . /app

# ─── Data volume ──────────────────────────────────────────────────────────────
# /data is mounted from host (docker-phishtrace/)
RUN mkdir -p /data

# ─── Environment defaults ────────────────────────────────────────────────────
ENV PHISHTRACE_MODE=micro \
    PHISHTRACE_DATA=/data \
    PHISHTRACE_PROXY="" \
    EXPERIMENT_INTERVAL=43200 \
    CRAWL_URL_INTERVAL=3600 \
    CRAWL_TRACE_INTERVAL=1800 \
    PYTHONPATH=/app

# ─── Entrypoint ──────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
