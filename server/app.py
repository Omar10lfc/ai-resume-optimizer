"""FastAPI backend replacing the Gradio UI (frontend is the Vite + React SPA).

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
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.responses import StreamingResponse
from starlette.status import (HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND,
                              HTTP_413_CONTENT_TOO_LARGE,
                              HTTP_429_TOO_MANY_REQUESTS)

from agent import config as agent_config
from agent import logging_config
from agent.llms import _groq_models_status
from agent.nodes.pdf_exporter import export_documents
from server import profile_store, run_store, runners

# ---------------------------------------------------------------------------
# Pipeline queue: Gradio-equivalent serialization (also the quota protector)
# ---------------------------------------------------------------------------

_QUEUE_MAX = 5          # max waiters, mirrors demo.queue(max_size=5)
_QUEUE_LEN = 0          # current waiters holding a queue slot
_QUEUE_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()   # one pipeline at a time

def _delete_output_root_dirs() -> int:
    """Deletes every directory under OUTPUT_ROOT (artifact/request dirs and
    the uploads dir) - Item 7 delete-everything. Best-effort; the 24h
    sweep remains the periodic backstop for strays."""
    import shutil
    root = agent_config.OUTPUT_ROOT
    if not root.is_dir():
        return 0
    removed = 0
    for entry in root.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed


# Item 8: per-app rate guard for the regenerate endpoint (app_id -> last attempt).
_REGEN_LIMITS: dict[int, float] = {}
_REGEN_LIMITS_LOCK = threading.Lock()
_REGEN_LIMIT_PER_APP_SECONDS = 60.0

_UPLOAD_SUFFIXES = {".pdf", ".docx"}
_ARTIFACT_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".tex": "text/x-tex",
    ".html": "text/html",
}

# Optional basic auth (both env vars required to enable, read once at import)
def _load_auth() -> tuple[str, str] | None:
    user = os.environ.get("APP_AUTH_USERNAME")
    password = os.environ.get("APP_AUTH_PASSWORD")
    return (user, password) if user and password else None


AUTH_CREDENTIALS = _load_auth()


# Header carrying a client's own Groq key (browser-extension BYOK support).
_CLIENT_GROQ_KEY_HEADER = "X-Groq-Api-Key"


def _client_groq_key(request: Request) -> str | None:
    """Reads (and sanity-caps) the optional client Groq key from a request."""
    key = (request.headers.get(_CLIENT_GROQ_KEY_HEADER) or "").strip()
    # Cap beyond any realistic key length: untrusted input should never reach
    # the LLM client constructor unchecked.
    return key[:512] if key else None


def _edge_limits() -> dict:
    """Item 8: HTTP-edge size limits + the pipeline deadline, surfaced so the
    UI can show users the allowed character counts before they hit a 413."""
    return {
        "resume_chars": agent_config.MAX_RESUME_CHARS,
        "job_chars": agent_config.MAX_JOB_CHARS,
        "notes_chars": agent_config.MAX_NOTES_CHARS,
        "pipeline_timeout_seconds": agent_config.MAX_PIPELINE_TIMEOUT_SECONDS,
    }


def _health_with_url_import(payload: dict) -> dict:
    """Add the URL-import configuration the UI needs to tell the truth.

    Health is also the config surface for the job-URL panel: whether import
    is enabled and which domains are allowed. No secrets - just the opt-in
    allowlist read at process start.
    """
    return {
        **payload,
        "url_import": {
            "enabled": bool(agent_config.ALLOWED_JOB_URL_DOMAINS),
            "allowed_domains": sorted(agent_config.ALLOWED_JOB_URL_DOMAINS),
        },
        "limits": _edge_limits(),
    }


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
    """Saves an UploadFile into the app-owned uploads dir with its suffix.

    Enforces the type gate and MAX_PDF_BYTES while copying. The caller owns
    cleanup (handled in _wrapped_pipeline's finally).
    """
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            "Unsupported resume file type (PDF or DOCX only).")
    max_bytes = agent_config.MAX_PDF_BYTES
    upload_dir = agent_config.OUTPUT_UPLOAD_DIR
    upload_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="resume_upload_",
                                      suffix=suffix, dir=str(upload_dir))
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
    # Item 8: HTTP-edge size caps - the pipeline already caps job_description
    # (runners._capped_job_description); cap the edge so oversized input is
    # rejected with 413 before any work starts.
    if len(job_description) > agent_config.MAX_JOB_CHARS:
        raise HTTPException(
            HTTP_413_CONTENT_TOO_LARGE,
            f"Job description exceeds the {agent_config.MAX_JOB_CHARS} "
            "character limit.")
    resume_path = _save_upload(resume_file) if resume_file else None
    if resume_path is None and not (resume_text or "").strip():
        raise HTTPException(
            HTTP_400_BAD_REQUEST,
            "Please upload a resume PDF or paste your resume text.")
    if resume_path is None and len(resume_text) > agent_config.MAX_RESUME_CHARS:
        raise HTTPException(
            HTTP_413_CONTENT_TOO_LARGE,
            f"Resume text exceeds the {agent_config.MAX_RESUME_CHARS} "
            "character limit.")
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
    logging_config.setup_logging()
    # Item 9: startup warning when APP_AUTH_* is unset on a non-local bind
    # (docker-compose.yml publishes on 127.0.0.1 by default and sets
    # APP_BIND_HOST=0.0.0.0 inside the container).
    if AUTH_CREDENTIALS is None:
        bind_host = (os.environ.get("APP_BIND_HOST")
                     or os.environ.get("HOST") or "127.0.0.1").strip().lower()
        if bind_host not in ("127.0.0.1", "localhost", "::1", ""):
            print("[Auth] WARNING: APP_AUTH_USERNAME/APP_AUTH_PASSWORD are not "
                  "set and the server is bound to a non-local address "
                  f"({bind_host}); anyone who can reach the port can spend the "
                  "model quota. Set APP_AUTH_* or bind to 127.0.0.1.")
    agent_config.init_output_dirs()
    agent_config.start_background_cleanup()
    # MR3: runs left 'running' by a previous process are dead (their mid-run
    # LLM work stopped at some checkpoint) - mark them failed-with-retry the
    # way Dossier's recoverInterruptedJobs() does, instead of letting a
    # silent band-aid resume stitch half-completed state. 'awaiting_notes'
    # rows survive: the review checkpoint is durable by design.
    recovered = run_store.recover_interrupted()
    if recovered:
        print(f"[Runs] Marked {recovered} interrupted run(s) as failed - retry from Step 2.")

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

    # Dev convenience for direct, non-proxy browser calls (Vite dev origin) and
    # for the browser-extension client (chrome-extension://<32-char id>).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_origin_regex=os.environ.get(
            "EXTENSION_ORIGINS", r"^chrome-extension://[a-p0-9]{32}$"),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def get_health(request: Request) -> dict:
        # BYOK clients verify THEIR key, not the server's (one cheap /models
        # call, zero generation tokens).
        user_key = _client_groq_key(request)
        if user_key:
            ok, message = _groq_models_status(user_key)
            return _health_with_url_import({"ok": ok, "message": message})
        health = getattr(request.app.state, "health", None)
        if health is None:
            return _health_with_url_import({
                "ok": None,
                "message": "Startup health check skipped (SKIP_STARTUP_HEALTHCHECK=1).",
            })
        return _health_with_url_import(health)

    @app.post("/api/v1/step1")
    def step1(request: Request,
              job_description: str = Form(...),
              resume_text: str = Form(""),
              resume_file: UploadFile | None = File(None),
              session: str = Form("")) -> StreamingResponse:
        job_description, resume_path = _validate_step_common(
            job_description, resume_text, resume_file)
        session = _safeguard_session(session)
        gen = runners.stream_step1(job_description,
                                   runners.resolve_resume_source(resume_path, resume_text),
                                   session, groq_api_key=_client_groq_key(request))
        temp = [resume_path] if resume_path else []
        return StreamingResponse(
            _wrapped_pipeline(gen, temp),
            media_type="text/event-stream", headers=_SSE_HEADERS)

    @app.post("/api/v1/step2")
    def step2(request: Request,
              job_description: str = Form(...),
              resume_text: str = Form(""),
              resume_file: UploadFile | None = File(None),
              notes: str = Form(""),
              session: str = Form(""),
              resume_theme: str = Form(""),
              include_diff: str = Query("")) -> StreamingResponse:
        # BB1 Phase B (PB3): include_diff=original asks the server to echo the
        # PARSED original resume in the result payload - DiffView then works
        # for uploads too (the SPA never parsed PDFs client-side).
        # resume_theme (round 14): the resume look (classic | serif |
        # modern); empty = the RESUME_THEME env knob, else the default.
        job_description, resume_path = _validate_step_common(
            job_description, resume_text, resume_file)
        # Item 8: HTTP-edge cap on the step-2 notes text.
        if len(notes or "") > agent_config.MAX_NOTES_CHARS:
            raise HTTPException(
                HTTP_413_CONTENT_TOO_LARGE,
                f"Notes exceed the {agent_config.MAX_NOTES_CHARS} character limit.")
        session = _safeguard_session(session)
        gen = runners.stream_step2(job_description,
                                   runners.resolve_resume_source(resume_path, resume_text),
                                   notes, session, groq_api_key=_client_groq_key(request),
                                   include_original=include_diff == "original",
                                   resume_theme=resume_theme)
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

    @app.get("/api/v1/runs/{session}")
    def get_run(session: str) -> dict:
        """MR3 reattach contract: the durable status of a session's latest
        run (status / last step / error / step-1 chips / the full result
        payload for done runs), so the SPA can re-render results after a
        reload or disconnect without re-running anything."""
        try:
            UUID(session)
        except ValueError:
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown session.") from None
        row = run_store.get(session)
        if row is None:
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown session.")
        return row

    # BB1 Phase A: the persistent profile store - application history with
    # their optimization attempts (auto-recorded on every completed run).

    @app.get("/api/v1/applications")
    def list_applications(limit: int = Query(default=50, ge=1, le=200)) -> dict:
        """Newest-first application summaries (no resume/job text - the
        detail endpoint carries it). Empty list when recording is off."""
        return {"applications": profile_store.list_applications(limit)}

    @app.get("/api/v1/applications/{app_id}")
    def get_application(app_id: int) -> dict:
        """One application: its evidence texts plus every recorded run
        (newest first, result payloads parsed) - the Fit history data."""
        row = profile_store.get_application(app_id)
        if row is None:
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown application.")
        return row

    @app.delete("/api/v1/applications/{app_id}")
    def delete_application(app_id: int) -> dict:
        if not profile_store.delete_application(app_id):
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown application.")
        return {"deleted": app_id}

    @app.delete("/api/v1/data")
    def delete_all_data() -> dict:
        """Item 7: 'delete everything' - applications + runs (profile,
        runs.sqlite3 ledger, checkpointer threads) + artifact dirs (WAL
        residue cleaned). Local trace copies (traces/, new-traces/,
        langsmith_export/) are NOT touched - they are audit exports, not
        app data. Requires auth when APP_AUTH_* is enabled."""
        from agent import graphs
        apps = profile_store.wipe()
        ledger = run_store.delete_all()
        threads = graphs.delete_all_checkpoint_threads()
        dirs = _delete_output_root_dirs()
        return {"applications": apps, "ledger_rows": ledger,
                "checkpoint_threads": threads, "artifact_dirs": dirs}

    # BB1 Phase B (PB2): history re-download. Stored run text outlives the
    # 24h artifact sweep, so regenerating from it is the honest "versions /
    # restore" for past runs. Regeneration runs through the CURRENT template
    # (may differ slightly from the original run's files); output lands in a
    # fresh request dir under OUTPUT_ROOT and is left for the regular sweep.
    _REGENERATE_KINDS = {
        "resume-pdf": "resume_pdf_path",
        "resume-docx": "resume_docx_path",
        "resume-tex": "resume_tex_path",
        "cover-pdf": "cover_letter_pdf_path",
    }

    @app.get("/api/v1/applications/{app_id}/runs/{run_id}/artifacts/{kind}")
    def regenerate_run_artifact(app_id: int, run_id: int, kind: str) -> FileResponse:
        export_key = _REGENERATE_KINDS.get(kind)
        if export_key is None:
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown artifact kind.")
        status, run = profile_store.get_run_for_app(app_id, run_id)
        if status == "missing_app":
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown application.")
        if status == "missing_run" or run is None:
            raise HTTPException(HTTP_404_NOT_FOUND, "Unknown run.")
        # Item 8: per-app rate guard on regenerate. Ownership above is
        # already validated (app + run); the guard limits repeated
        # regenerate spam to one attempt per app per minute.
        with _REGEN_LIMITS_LOCK:
            last = _REGEN_LIMITS.get(app_id, 0.0)
            if time.time() - last < _REGEN_LIMIT_PER_APP_SECONDS:
                raise HTTPException(
                    HTTP_429_TOO_MANY_REQUESTS,
                    "Regenerate rate limit exceeded - try again shortly.")
            _REGEN_LIMITS[app_id] = time.time()
        result = run.get("result") or {}
        optimized = result.get("optimized_resume") or ""
        cover = result.get("cover_letter") or ""
        # _extract_results' placeholder strings are not exportable text.
        if optimized == "No resume generated.":
            optimized = ""
        if cover == "No cover letter generated.":
            cover = ""
        needed_text = optimized if kind.startswith("resume-") else cover
        if not needed_text.strip():
            raise HTTPException(
                HTTP_404_NOT_FOUND,
                "Stored run text not available for regeneration.")
        request_dir = agent_config.OUTPUT_ROOT / str(uuid4())
        exported = export_documents(optimized, cover, request_dir)
        path = Path(exported.get(export_key) or "")
        if not path.is_file():
            raise HTTPException(
                HTTP_404_NOT_FOUND, "Artifact could not be regenerated.")
        content_type = _ARTIFACT_CONTENT_TYPES.get(
            path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=content_type,
                            filename=path.name)

    frontend_dist = (Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()
    if frontend_dist.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()


__all__ = ["app", "create_app", "require_auth", "runners"]
