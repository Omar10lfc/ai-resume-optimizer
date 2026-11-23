"""Environment setup, model constants, output directories, and tuning knobs."""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- LLM provider (round 15, roadmap item 13; env-overridable) ---
# LLM_PROVIDER selects the chat-model family: groq (default, the historical
# engine) | openai | anthropic | openai-compatible (ANY OpenAI-wire endpoint:
# a local Ollama/LM Studio/vLLM server - the PII-on-device posture, the raw
# JD + resume never leave the machine). GROQ_API_KEY stays the UNIVERSAL
# API-key slot for every provider (historical name); LLM_BASE_URL overrides
# the endpoint when set (for anthropic, an anthropic-compatible endpoint).
_LLM_PROVIDER_RAW = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
LLM_PROVIDER = (_LLM_PROVIDER_RAW
                if _LLM_PROVIDER_RAW in {"groq", "openai", "anthropic",
                                         "openai-compatible"}
                else "groq")

# Per-provider tier defaults: (primary, fallback, emergency) - the classic
# three-tier ladder kept WITHIN one provider (a degraded run falls to a
# weaker model of the same vendor, never a silent cross-vendor hop).
# PRIMARY_MODEL / FALLBACK_MODEL / EMERGENCY_MODEL env overrides win over
# these for every provider.
_PROVIDER_MODEL_DEFAULTS = {
    # 'llama-3.1-8b-instant' was decommissioned from the Groq catalog
    # (Nov 10 2026 - live 404 'model_not_found' once a run fell through
    # primary+fallback). Verified present on the live catalog:
    # 'qwen/qwen3.8-27b'.
    "groq": ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"),
    "openai": ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"),
    "anthropic": ("claude-sonnet-4-5", "claude-haiku-4-5",
                  "claude-3-5-haiku-latest"),
    # A self-hosted endpoint serves whatever models it has - set
    # PRIMARY_MODEL/FALLBACK_MODEL/EMERGENCY_MODEL to its names (Ollama's
    # gpt-oss tags assumed as the default; the health check lists what the
    # endpoint serves).
    "openai-compatible": ("gpt-oss:120b", "gpt-oss:20b", "gpt-oss:20b"),
}
_default_primary, _default_fallback, _default_emergency = (
    _PROVIDER_MODEL_DEFAULTS.get(LLM_PROVIDER,
                                 _PROVIDER_MODEL_DEFAULTS["groq"]))
_PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", _default_primary)
_FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", _default_fallback)
_EMERGENCY_MODEL = os.environ.get("EMERGENCY_MODEL", _default_emergency)

# --- MR5 provider abstraction (env-overridable) ---
# LLM_BASE_URL: any OpenAI-compatible endpoint (Ollama http://localhost:11434/v1,
# LM Studio http://localhost:1234/v1, OpenRouter https://openrouter.ai/api/v1, ...).
# Empty = the provider's official API. When set, the chat models
# are pointed at that endpoint (the groq/openai-compatible classes speak the
# OpenAI wire protocol - proven by the MR4 wire tests against the fake server)
# and the Groq-specific reasoning_effort parameter is omitted. GROQ_API_KEY
# doubles as the generic API-key slot; a custom endpoint that needs no auth
# (local Ollama) runs without one, which also relaxes the import-time key
# requirement below.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip()
# Hard cap per LLM call in seconds (the SDK's own default is 10 minutes - a hung
# call used to stall the serialized pipeline for everyone).
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", 120))
# Output-token cap for providers that require one explicitly: langchain-
# anthropic leaves max_tokens None, which the API then defaults to a small
# value that truncates full resumes. Generous default; env-overridable.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 8192))

# --- Resume renderer (BB2, env-overridable; Nov 15 2026: auto is
# look-driven - LaTeX prints the default again) ---
# auto (default): the classic look (the runner's canonical fill and any
# explicit classic pick) prints through the bundled LaTeX template - the
# historical engine; serif/modern looks route to the HTML/CSS theme
# printed by the system's headless Chromium/Edge when one is found, and
# to LaTeX otherwise. html: the HTML theme for EVERY look, falling back
# to LaTeX when no browser is available or the print fails. latex: the
# historical LaTeX path for every look, unchanged.
_RESUME_RENDERER_RAW = os.environ.get("RESUME_RENDERER", "auto").strip().lower()
RESUME_RENDERER = (_RESUME_RENDERER_RAW
                   if _RESUME_RENDERER_RAW in {"auto", "html", "latex"}
                   else "auto")
# Explicit override for the print engine binary; empty = auto-discovery
# (PATH, then the standard Chrome/Edge install locations, then a Playwright
# browser cache).
RENDER_BROWSER_PATH = os.environ.get("RENDER_BROWSER_PATH", "").strip()
# Resume look knob (round 14): a pure CSS-variable set over fonts, colors,
# and heading decoration - NEVER sizes (the density variables stay the only
# size authority, so the measured tier ladder holds for every theme).
# Duplicated from html_render.THEMES (which this module's consumer imports)
# to avoid the circular import; "" = pipeline default (classic). The
# per-request value from the step-2 form overrides this env fallback.
_RESUME_THEME_RAW = os.environ.get("RESUME_THEME", "").strip().lower()
RESUME_THEME = (_RESUME_THEME_RAW
                if _RESUME_THEME_RAW in {"classic", "serif", "modern"}
                else "")
# Optional PDF layout diagnostics (agent/verify_layout.py): Poppler
# `pdftotext -bbox` measurement of the shipped resume/cover PDFs (holes,
# orphaned entry headers, footer collisions, thin final pages). Off by
# default - a diagnostic, not part of the render contract. Set
# LAYOUT_VERIFY_ENABLED=1 to include the advisory rows in the Integrity tab.
LAYOUT_VERIFY_ENABLED = os.environ.get(
    "LAYOUT_VERIFY_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}

# --- Output Directories & Persistence ---# BB1 Phase B (PB1): artifacts live under the project's gitignored data/
# dir by default (visible to the user, survives OS temp cleaners, and
# ships with the workspace instead of vanishing into %TEMP%). Env-
# overridable - point OUTPUT_ROOT at a scratch disk or the historical
# %TEMP% location if preferred. Tests monkeypatch the attribute directly.
_OUTPUT_ROOT_RAW = os.environ.get("OUTPUT_ROOT", "").strip()
OUTPUT_ROOT = (Path(_OUTPUT_ROOT_RAW) if _OUTPUT_ROOT_RAW
               else Path(__file__).resolve().parents[1] / "data" / "artifacts")
OUTPUT_UPLOAD_DIR = OUTPUT_ROOT / "uploads"
OUTPUT_RETENTION_HOURS = 24
# Durable session checkpoints (the Dossier study's crash-resilient job store,
# consumed by graphs._create_checkpointer's SqliteSaver hook): DEFAULTS ON
# under the gitignored data/ dir, so a server restart no longer loses step
# state (the Nov 12-13 forensics gap: run-time notes payloads were
# unrecoverable with the MemorySaver default). Escape hatches:
# CHECKPOINTER_DB_PATH=off (or none/memory) restores the zero-configuration
# in-memory checkpointer (tests pin this in tests/conftest.py); any other
# value is used as the SQLite path as-is.
def _resolve_checkpointer_db_path(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.lower() in {"off", "none", "memory"}:
        return ""
    return raw or str(Path(__file__).resolve().parents[1] / "data"
                      / "checkpoints" / "checkpoints.sqlite3")


CHECKPOINTER_DB_PATH = _resolve_checkpointer_db_path(
    os.environ.get("CHECKPOINTER_DB_PATH", ""))
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
# Item 8: edge caps + the pipeline-level deadline.
MAX_NOTES_CHARS = int(os.environ.get("MAX_NOTES_CHARS", 4_000))
MAX_PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("MAX_PIPELINE_TIMEOUT_SECONDS", 900))
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


def _sweep_orphaned_uploads(max_age_hours: int = OUTPUT_RETENTION_HOURS) -> int:
    """Deletes old app-owned upload files older than the retention window."""
    import time
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    try:
        if not OUTPUT_UPLOAD_DIR.is_dir():
            return 0
        for entry in OUTPUT_UPLOAD_DIR.iterdir():
            try:
                if (entry.is_file() and entry.name.startswith("resume_upload_")
                        and entry.stat().st_mtime < cutoff):
                    entry.unlink()
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
    OUTPUT_UPLOAD_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    _cleanup_stale_request_dirs()
    _sweep_orphaned_uploads()


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
