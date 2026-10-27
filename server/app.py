"""FastAPI backend replacing the Gradio UI (frontend is now Next.js + BFF).

Ports the Gradio app's startup block and runtime guarantees 1:1:

- lifespan: ``init_output_dirs()`` + ``start_background_cleanup()`` + the
  optional Groq startup health check (``SKIP_STARTUP_HEALTHCHECK=1`` opts out).
- Pipeline serialization: one pipeline at a time (concurrency would
  double-spend Groq quota) with a queue capped at 5 waiters - the exact
  semantics of the old ``demo.queue(max_size=5, default_concurrency_limit=1)``.
  Wait is streamed to the client as a status event, mirroring Gradio's queue.
- Optional basic auth for public deployments (``APP_AUTH_USERNAME`` /
  ``APP_AUTH_PASSWORD``, both required to enable) - replaces
  ``GRADIO_AUTH_USERNAME``/``GRADIO_AUTH_PASSWORD``, which previously gated
  the app so anyone reaching it could not spend the quota.

Transport notes:
- Step 1 / Step 2 accept the same multipart inputs as the Gradio version
  (job description + resume file OR pasted text) and respond with a
  ``text/event-stream`` whose events are exactly the runner dict events
  (status / warning / error / done / result).
- Artifact downloads validate the session id (UUID), restrict the file to
  a gated allowlist, and resolve strictly inside OUTPUT_ROOT.
"""

import json
import os
import secrets
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_413_CONTENT_TOO_LARGE

from agent import config as agent_config
from server import runners

# ---------------------------------------------------------------------------
# Pipeline queue: Gradio-equivalent serialization (also the quota protector)
# ---------------------------------------------------------------------------

_QUEUE_MAX = 5          # max waiters, mirrors demo.queue(max_size=5)
_QUEUE_LEN = 0          # current waiters holding a queue slot
_QUEUE_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()   # one pipeline at a time

_UPLOAD_SUFFIXES = {".pdf", ".docx"}
_ARTIFACT_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".tex": "text/x-tex",
}

# Optional basic auth (both env vars required to enable, read once at import)
def _load_auth() -> tuple[str, str] | None:
    user = os.environ.get("APP_AUTH_USERNAME")
    password = os.environ.get("APP_AUTH_PASSWORD")
    return (user, password) if user and password else None


AUTH_CREDENTIALS = _load_auth()


_security = HTTPBasic(auto_error=False)


def require_auth(
    credentials: HTTPBasicCredentials | None = Depends(_security),
) -> None:
    """Basic auth dependency - open access when APP_AUTH_* is not configured."""
    if AUTH_CREDENTIALS is None:
        return
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic realm=resume-optimizer"},
        )
    user, password = AUTH_CREDENTIALS
    user_ok = secrets.compare_digest(credentials.username.encode("utf-8"),
                                     user.encode("utf-8"))
    pass_ok = secrets.compare_digest(credentials.password.encode("utf-8"),
                                     password.encode("utf-8"))
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic realm=resume-optimizer"},
        )


def _sse_line(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _wrapped_pipeline(gen, temp_paths: list[str]):
    """Queues one pipeline run and cleans up temp upload files afterwards.

    Queue semantics mirror the retired Gradio queue: max 5 waiters, runs
    strictly one at a time. Client-visible wait happens as regular status
    events. On disconnect mid-stream the generator is closed: the finally
    blocks release the run lock and delete temp files (the pipeline's LLM
    spend, once started, completes as before).
    """
    global _QUEUE_LEN
    queued = False
    try:
        with _QUEUE_LOCK:
            if _QUEUE_LEN >= _QUEUE_MAX:
                full = True
            else:
                _QUEUE_LEN += 1
                queued = True
                full = False
        if full:
            yield _sse_line("error", {
                "type": "error",
                "message": f"Pipeline queue is full ({_QUEUE_MAX} waiting). Try again shortly.",
            })
            return

        if _RUN_LOCK.locked():
            with _QUEUE_LOCK:
                waiting = _QUEUE_LEN
            ahead = waiting - 1
            wait_msg = ("Waiting for a free pipeline slot..." if ahead == 0
                        else f"Waiting for a free pipeline slot ({ahead} ahead of you)...")
            yield _sse_line("status", {"type": "status", "message": wait_msg})

        _RUN_LOCK.acquire()
        try:
            yield from (_sse_line(e["type"], e) for e in gen)
        finally:
            _RUN_LOCK.release()
    finally:
        if queued:
            with _QUEUE_LOCK:
                _QUEUE_LEN = max(0, _QUEUE_LEN - 1)
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # disable proxy buffering for live streaming
}


def _save_upload(upload: UploadFile) -> str:
    """Saves an UploadFile to a named temp file with its original suffix.

    Enforces the type gate and MAX_PDF_BYTES while copying. The caller owns
    cleanup (handled in _wrapped_pipeline's finally).
    """
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            "Unsupported resume file type (PDF or DOCX only).")
    max_bytes = agent_config.MAX_PDF_BYTES
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="resume_upload_",
                                      suffix=suffix)
    try:
        written = 0
        with tmp:
            while chunk := upload.file.read(64 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        HTTP_413_CONTENT_TOO_LARGE,
                        f"Resume file exceeds the {max_bytes // (1024 * 1024)} MB limit.")
                tmp.write(chunk)
        return tmp.name
    except BaseException:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def _validate_step_common(job_description: str, resume_text: str,
                          resume_file: UploadFile | None) -> tuple[str, str | None]:
    """Validation ported from the Gradio step handlers (HTTP 400s)."""
    if not job_description or not job_description.strip():
        raise HTTPException(HTTP_400_BAD_REQUEST,
                            "Please provide a Job Description (text or URL).")
    resume_path = _save_upload(resume_file) if resume_file else None
    if resume_path is None and not (resume_text or "").strip():
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            "Please upload a resume PDF or paste your resume text.")
    return job_description, resume_path


def _safeguard_session(session: str | None) -> str | None:
    """Session ids are UUIDs; anything else is a client bug -> clean 400."""
    if session is None or session == "":
        return None
    try:
        return str(UUID(session))
    except ValueError as exc:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Invalid session id.") from exc


# ---------------------------------------------------------------------------
# App factory / lifespan (ports app.py's __main__ startup block)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_config.init_output_dirs()
    agent_config.start_background_cleanup()

    # Fail fast at startup with an actionable message instead of letting users
    # hit a confusing provider error deep inside Step 2. One cheap /models
    # call (zero generation tokens).
    health = None
    if os.environ.get("SKIP_STARTUP_HEALTHCHECK", "").lower() != "1":
        from agent.llms import check_provider_health
        ok, message = check_provider_health()
        print(f"[Health] Groq: {'OK' if ok else 'PROBLEM'} - {message}")
        if not ok:
            print("[Health] Launching anyway - pipeline runs will fail until this is resolved.")
        health = {"ok": ok, "message": message}
    app.state.health = health
    yield


def create_app() -> FastAPI:
    if AUTH_CREDENTIALS is None:
        print("[Auth] No APP_AUTH_USERNAME/APP_AUTH_PASSWORD set - app is unauthenticated (fine for localhost).")
    # docs/schema routes disabled: FastAPI's built-in docs do NOT inherit
    # app-level auth dependencies, so they would expose the API schema
    # unauthenticated. The Next.js BFF owns the public contract instead.
    app = FastAPI(title="AI Resume Optimizer Agent API",
                  version="2.0.0",
                  docs_url=None,
                  redoc_url=None,
                  openapi_url=None,
                  lifespan=lifespan,
                  dependencies=[Depends(require_auth)])

    # Dev convenience for direct, non-BFF calls; the Next.js BFF proxies
    # server-side (same origin), so CORS is not needed in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def get_health(request: Request) -> dict:
        health = getattr(request.app.state, "health", None)
        if health is None:
            return {"ok": None,
                    "message": "Startup health check skipped (SKIP_STARTUP_HEALTHCHECK=1)."}
        return health

    @app.post("/api/v1/step1")
    def step1(job_description: str = Form(...),
              resume_text: str = Form(""),
              resume_file: UploadFile | None = File(None),
              session: str = Form("")) -> StreamingResponse:
        job_description, resume_path = _validate_step_common(
            job_description, resume_text, resume_file)
        session = _safeguard_session(session)
        gen = runners.stream_step1(job_description,
                                   runners.resolve_resume_source(resume_path, resume_text),
                                   session)
        temp = [resume_path] if resume_path else []
        return StreamingResponse(
            _wrapped_pipeline(gen, temp),
            media_type="text/event-stream", headers=_SSE_HEADERS)

    @app.post("/api/v1/step2")
    def step2(job_description: str = Form(...),
              resume_text: str = Form(""),
              resume_file: UploadFile | None = File(None),
              notes: str = Form(""),
              session: str = Form("")) -> StreamingResponse:
        job_description, resume_path = _validate_step_common(
            job_description, resume_text, resume_file)
        session = _safeguard_session(session)
        gen = runners.stream_step2(job_description,
                                   runners.resolve_resume_source(resume_path, resume_text),
                                   notes, session)
        temp = [resume_path] if resume_path else []
        return StreamingResponse(
            _wrapped_pipeline(gen, temp),
            media_type="text/event-stream", headers=_SSE_HEADERS)

    @app.get("/api/v1/artifacts/{session}/{kind}")
    def get_artifact(session: str, kind: str, file: str = Query(...)) -> FileResponse:
        try:
            UUID(session)
        except ValueError:
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown session.") from None
        if kind not in runners._ARTIFACT_KINDS:
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown artifact kind.")
        base = (agent_config.OUTPUT_ROOT / session).resolve()
        candidate = (base / Path(file).name).resolve()
        if (not base.is_dir() or candidate.parent != base
                or not candidate.is_file()):
            raise HTTPException(HTTP_404_NOT_FOUND, "Artifact not available.")
        content_type = _ARTIFACT_CONTENT_TYPES.get(
            candidate.suffix.lower(), "application/octet-stream")
        return FileResponse(candidate, media_type=content_type,
                            filename=candidate.name)

    frontend_dist = (Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()
    if frontend_dist.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()


__all__ = ["app", "create_app", "require_auth", "runners"]
