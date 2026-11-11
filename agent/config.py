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
# 'llama-3.1-8b-instant' was decommissioned from the Groq catalog (Nov 10 2026 -
# live 404 'model_not_found' once a run fell through primary+fallback).
# Verified present on the live catalog: 'qwen/qwen3.8-27b'.
_EMERGENCY_MODEL = os.environ.get("EMERGENCY_MODEL", "qwen/qwen3.8-27b")

# --- MR5 provider abstraction (env-overridable) ---
# LLM_BASE_URL: any OpenAI-compatible endpoint (Ollama http://localhost:11434/v1,
# LM Studio http://localhost:1234/v1, OpenRouter https://openrouter.ai/api/v1, ...).
# Empty = Groq's official API (the historical default). When set, the chat models
# are pointed at that endpoint (ChatGroq speaks the OpenAI wire protocol - proven
# by the MR4 wire tests against the fake server) and the Groq-specific
# reasoning_effort parameter is omitted. GROQ_API_KEY doubles as the generic
# API-key slot; a custom endpoint that needs no auth (local Ollama) runs without
# one, which also relaxes the import-time key requirement below.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip()
# Hard cap per LLM call in seconds (the SDK's own default is 10 minutes - a hung
# call used to stall the serialized pipeline for everyone).
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", 120))

# --- Output Directories & Persistence ---
OUTPUT_ROOT = Path(tempfile.gettempdir()) / "resume_optimizer_output"
OUTPUT_RETENTION_HOURS = 24
CHECKPOINTER_DB_PATH = os.environ.get("CHECKPOINTER_DB_PATH", "").strip()
# MR3 durable runs: SQLite ledger of run status/results (per-step progress,
# step-1 chips for mid-pipeline recovery, done-run result payloads for
# reattach). Defaults next to the artifacts; env-overridable (tests point it
# at a tmp file). The artifact cleanup only removes stale DIRECTORIES under
# OUTPUT_ROOT, so the ledger file itself survives the sweep.
RUNS_DB_PATH = (os.environ.get("RUNS_DB_PATH", "").strip()
                or str(OUTPUT_ROOT / "runs.sqlite3"))
# BB1 persistent profile store: applications + their optimization attempts
# (scores, requirement tables, full result payloads). Unlike the runs ledger
# (24h retention, next to the artifacts), history is PERSISTENT by design -
# the resume/job text at rest is the point (the Dossier study's Application
# record: every artifact derives from it). Defaults under the project root's
# gitignored data/ dir; set PROFILE_DB_PATH=off to disable recording (no
# file is created, no history is written).
_PROFILE_DB_RAW = os.environ.get("PROFILE_DB_PATH", "").strip()
PROFILE_DB_ENABLED = _PROFILE_DB_RAW.lower() != "off"
PROFILE_DB_PATH = (_PROFILE_DB_RAW
                   or str(Path(__file__).resolve().parents[1] / "data" / "profile.sqlite3"))

# --- Input Limits (env-overridable) ---
MAX_PDF_BYTES = int(os.environ.get("MAX_PDF_BYTES", 5 * 1024 * 1024))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", 8))
MAX_RESUME_CHARS = int(os.environ.get("MAX_RESUME_CHARS", 12_000))
MAX_JOB_CHARS = int(os.environ.get("MAX_JOB_CHARS", 6_000))
MAX_FETCH_BYTES = int(os.environ.get("MAX_FETCH_BYTES", 1_000_000))
FETCH_TIMEOUT_SECONDS = int(os.environ.get("FETCH_TIMEOUT_SECONDS", 8))

# --- Self-correction loop tuning (env-overridable) ---
SCORE_THRESHOLD = int(os.environ.get("SCORE_THRESHOLD", 80))
# 2 = one polish pass applying the reviewer's feedback beyond the first draft.
# The gate routes there on BOTH outcomes the reviewer produces: failing
# composites (score < 80) and passing ones that still carry feedback (Nov 6
# 2026 directive: passing runs were dropping the reviewer's concrete
# suggestion unapplied). The ats_check best-of guard keeps the best draft, so
# a second pass can never regress the result, and the cap bounds the cost.
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 2))
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


def start_background_cleanup(
    interval_seconds: int = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", 21600)),
    max_age_hours: int = OUTPUT_RETENTION_HOURS,
):
    """Starts a daemon thread that periodically purges stale output directories.

    Runs in the background every `interval_seconds` (default: 6 hours) to prevent
    unbounded disk growth in continuous 24/7 production deployments.
    """
    import threading
    import time
    import logging

    log = logging.getLogger(__name__)

    def _cleanup_worker():
        while True:
            time.sleep(interval_seconds)
            try:
                removed = _cleanup_stale_request_dirs(max_age_hours=max_age_hours)
                if removed > 0:
                    log.info(f"[Cleanup] Removed {removed} stale request directory/ies.")
            except Exception as e:
                log.warning(f"[Cleanup] Background cleanup failed: {e}")

    thread = threading.Thread(target=_cleanup_worker, name="DiskCleanupDaemon", daemon=True)
    thread.start()
    return thread


if not os.environ.get("GROQ_API_KEY") and not LLM_BASE_URL:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Please set it in your .env file or as an environment variable.\n"
        "Get a free key at: https://console.groq.com/keys\n"
        "(Or set LLM_BASE_URL to an OpenAI-compatible endpoint that needs no key,"
        " e.g. a local Ollama/LM Studio server.)"
    )
