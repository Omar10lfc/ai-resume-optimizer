"""NODE 0: Loader - loads PDF or web data."""
from pathlib import Path
from uuid import uuid4

from ..config import OUTPUT_ROOT, MAX_JOB_CHARS, MAX_RESUME_CHARS
from ..state import AgentState
from ..helpers import _safe_print, _safe_truncate
from ..tracing import register_known_name, candidate_name_hint
from ..url_fetch import _fetch_job_description, _load_resume_docx, _load_resume_pdf


def _normalize_raw_text(text: str) -> str:
    if not text:
        return ""
    return text.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\t', '\t')


def loader_node(state: AgentState):
    _safe_print(f"\n--- NODE 0: LOADING DATA ---")

    request_id = state.get("request_id") or str(uuid4())
    request_dir = OUTPUT_ROOT / request_id

    job_content = state['job_description']
    if state['job_description'].strip().startswith("http"):
        try:
            job_content = _fetch_job_description(state['job_description'])
        except Exception as e:
            raise ValueError(f"Could not import job URL: {e}") from e
    else:
        job_content = _safe_truncate(job_content, MAX_JOB_CHARS, "Job description")

    resume_source = state['original_resume']
    suffix = Path(resume_source).suffix.lower()
    if suffix == ".pdf" and Path(resume_source).exists():
        try:
            resume_content = _load_resume_pdf(resume_source)
        except Exception as e:
            raise ValueError(f"Could not load resume PDF: {e}") from e
    elif suffix == ".docx" and Path(resume_source).exists():
        try:
            resume_content = _load_resume_docx(resume_source)
        except Exception as e:
            raise ValueError(f"Could not load resume DOCX: {e}") from e
    else:
        resume_content = _safe_truncate(resume_source, MAX_RESUME_CHARS, "Resume")

    job_content = _normalize_raw_text(job_content)
    resume_content = _normalize_raw_text(resume_content)

    # PII hygiene (production Item 4): register the candidate name parsed from
    # the resume so name masking covers PDF/DOCX payloads. Must precede every
    # later run's tracing; the loader's own input may already be masked by an
    # earlier app-level pre-registration for pasted resumes.
    register_known_name(candidate_name_hint(resume_content))

    return {
        "job_text": job_content,
        "resume_text": resume_content,
        "request_id": request_id,
        "output_dir": str(request_dir),
    }
