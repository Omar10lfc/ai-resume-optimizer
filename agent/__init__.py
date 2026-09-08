"""Agent package: public API exports for app.py and tests."""
from .state import AgentState, ReviewOutput, SemanticVerdict, SemanticReview, SYSTEM_GUARDRAIL, _untrusted
from .helpers import _force_ascii, _strip_emoji_for_pdf, _strip_emoji_from_html, _strip_code_fences, _filter_hallucinated_sections, _sanitize_html_for_pdf, _html_to_pdf
from .ats import _extract_keywords, _simple_stem, _keyword_found_in_text, compute_ats_match, _job_brief, _boundary_contains, freq_ok
from .url_fetch import _validate_job_url, _is_public_host, _fetch_job_description, _load_resume_pdf
from .latex_render import (
    parse_markdown_resume, render_resume_tex, latex_to_pdf, latex_to_docx,
    HAVE_PANDOC, HAVE_PDFLATEX,
)
from .config import (
    SCORE_THRESHOLD, MAX_ITERATIONS, MAX_PDF_PAGES, MAX_RESUME_CHARS,
    MAX_JOB_CHARS, MAX_PDF_BYTES, ATS_SCORE_WEIGHT, LLM_SCORE_WEIGHT,
    OUTPUT_ROOT, FETCH_TIMEOUT_SECONDS, ALLOWED_JOB_URL_DOMAINS, MAX_FETCH_BYTES,
    _EMERGENCY_MODEL, _cleanup_stale_request_dirs,
)
from .tracing import TRACE_CALLBACKS
from .llms import llm_strict, llm_creative, llm_fast, _RETRYABLE_EXCEPTIONS
from .nodes import (
    loader_node, scanner_node, improver_node, reviewer_node,
    ats_check_node, cover_letter_node, interview_prep_node, pdf_exporter_node,
)
from .graphs import full_app, agent_app, should_continue

__all__ = [
    # State & models
    "AgentState", "ReviewOutput", "SemanticVerdict", "SemanticReview",
    "SYSTEM_GUARDRAIL", "_untrusted",
    # Helpers
    "_force_ascii", "_strip_emoji_for_pdf", "_strip_emoji_from_html", "_strip_code_fences",
    "_filter_hallucinated_sections", "_sanitize_html_for_pdf",
    # ATS engine
    "_extract_keywords", "_simple_stem", "compute_ats_match", "_job_brief",
    "_boundary_contains", "freq_ok",
    # Config
    "SCORE_THRESHOLD", "MAX_ITERATIONS", "MAX_PDF_PAGES", "MAX_RESUME_CHARS",
    "MAX_JOB_CHARS", "MAX_PDF_BYTES", "ATS_SCORE_WEIGHT", "LLM_SCORE_WEIGHT",
    "OUTPUT_ROOT", "FETCH_TIMEOUT_SECONDS", "ALLOWED_JOB_URL_DOMAINS", "MAX_FETCH_BYTES",
    "_EMERGENCY_MODEL",
    # Tracing
    "TRACE_CALLBACKS",
    # LLMs
    "llm_strict", "llm_creative", "llm_fast", "_RETRYABLE_EXCEPTIONS",
    # Nodes
    "loader_node", "scanner_node", "improver_node", "reviewer_node",
    "ats_check_node", "cover_letter_node", "interview_prep_node", "pdf_exporter_node",
    # LaTeX rendering
    "parse_markdown_resume", "render_resume_tex", "latex_to_pdf", "latex_to_docx",
    "HAVE_PANDOC", "HAVE_PDFLATEX",
    # Graphs
    "full_app", "agent_app", "should_continue",
]
