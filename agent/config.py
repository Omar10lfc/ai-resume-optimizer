"""Environment setup, model constants, output directories, and tuning knobs."""
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration (env-overridable) ---
_PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "openai/gpt-oss-120b")
_FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "openai/gpt-oss-20b")
_EMERGENCY_MODEL = os.environ.get("EMERGENCY_MODEL", "llama-3.1-8b-instant")

# --- Output Directories ---
OUTPUT_ROOT = Path(tempfile.gettempdir()) / "resume_optimizer_output"
OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)

OUTPUT_RETENTION_HOURS = 24

# --- Input Limits (env-overridable) ---
MAX_PDF_BYTES = int(os.environ.get("MAX_PDF_BYTES", 5 * 1024 * 1024))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", 8))
MAX_RESUME_CHARS = int(os.environ.get("MAX_RESUME_CHARS", 12_000))
MAX_JOB_CHARS = int(os.environ.get("MAX_JOB_CHARS", 6_000))
MAX_FETCH_BYTES = int(os.environ.get("MAX_FETCH_BYTES", 1_000_000))
FETCH_TIMEOUT_SECONDS = int(os.environ.get("FETCH_TIMEOUT_SECONDS", 8))

# --- Self-correction loop tuning (env-overridable) ---
SCORE_THRESHOLD = int(os.environ.get("SCORE_THRESHOLD", 80))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 1))
ATS_SCORE_WEIGHT = float(os.environ.get("ATS_SCORE_WEIGHT", 0.6))
LLM_SCORE_WEIGHT = float(os.environ.get("LLM_SCORE_WEIGHT", 0.4))

# --- URL import (opt-in) ---
ALLOWED_JOB_URL_DOMAINS = {
    domain.strip().lower()
    for domain in os.environ.get("ALLOWED_JOB_URL_DOMAINS", "").split(",")
    if domain.strip()
}


def _cleanup_stale_request_dirs(max_age_hours: int = OUTPUT_RETENTION_HOURS) -> int:
    """Deletes per-request output directories older than the retention window."""
    import time
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    try:
        for entry in OUTPUT_ROOT.iterdir():
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


# No longer runs at import time — call init_output_dirs() explicitly from app/graph entry points.
# This prevents filesystem side effects during test collection.
def init_output_dirs():
    """Initialize output directories and clean up stale request dirs.

    Call once at application startup (app.py main block or graphs.py CLI).
    """
    OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    _cleanup_stale_request_dirs()


if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Please set it in your .env file or as an environment variable.\n"
        "Get a free key at: https://console.groq.com/keys"
    )
