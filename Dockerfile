# Job Application Optimizer - single-container Docker (Item 9)
#
# Build & run (compose is the primary path):
#   docker compose up -d
#   docker build -t job-application-optimizer .      # optional, compose uses this tag
#
# Item 9 hardening notes:
#   - Non-root user (app, uid 1000 by default; override the compose `user:`
#     line with HOST_UID/HOST_GID when the host uid differs).
#   - Chromium runs sandboxed by default (RENDER_NO_SANDBOX stays off).
#   - TeX Live with openin_any=p / openout_any=p set in the image (the
#     app-level guard in agent/latex_render.py is the backstop - MiKTeX
#     ignores those env vars).
#   - Versions are pinned: base tag below (recommended: pin to a digest),
#     apt packages pinned by the Debian 12 base.
#   - Health check: docker-compose.yml.
#   - Startup warning: server/app.py warns when APP_AUTH_* is unset on a
#     non-local bind (docker-compose.yml sets APP_BIND_HOST=0.0.0.0).

# Recommended for production: pin to a digest.
FROM python:3.12-slim-bookworm

# --- non-root user + system dependencies (TeX Live + Chromium + fonts) ---
RUN useradd --uid 1000 --create-home --shell /bin/bash app

RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        fonts-noto-color-emoji \
        lmodern \
        chromium \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# TeX Live paranoid file access (Item 9): never read/write outside the
# compile dir.
ENV openin_any=p openout_any=p

WORKDIR /app

# Python deps first (build cache layer), then the app code.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
