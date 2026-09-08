import os
import gradio as gr
from uuid import uuid4
from agent import agent_app, TRACE_CALLBACKS
from agent.config import init_output_dirs

# Redacted-tracing callbacks go with every invoke (empty list = tracing off)
_INVOKE_CONFIG_BASE = {"callbacks": TRACE_CALLBACKS}

# Per-node status messages for the streaming pipeline view
NODE_STATUS = {
    "loader": "Loading documents...",
    "scanner": "Scanning for skill gaps...",
    "improver": "Improving resume...",
    "reviewer": "Reviewer is grading the draft...",
    "ats_check": "Computing ATS keyword match...",
    "cover_letter": "Writing cover letter...",
    "interview_prep": "Preparing interview questions...",
    "pdf_exporter": "Exporting PDFs...",
}


def _thread_config(thread_id: str) -> dict:
    cfg = dict(_INVOKE_CONFIG_BASE)
    cfg["configurable"] = {"thread_id": thread_id}
    return cfg

# ============================================================
# CUSTOM CSS — Refined dark theme with glassmorphism
# ============================================================
CUSTOM_CSS = """
/* ---- Global ---- */
.gradio-container {
    max-width: 100% !important;
    padding: 0 1.5rem !important;
}

/* ---- Hero Section ---- */
.hero-section {
    text-align: center;
    padding: 2.5rem 2rem 2rem;
    background: linear-gradient(135deg,
        rgba(56, 189, 248, 0.07) 0%,
        rgba(14, 165, 233, 0.04) 50%,
        rgba(125, 211, 252, 0.03) 100%);
    border-radius: 16px;
    border: 1px solid rgba(56, 189, 248, 0.12);
    margin-bottom: 2rem !important;
}

.hero-section .hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #e0f2fe !important;
    -webkit-text-fill-color: unset !important;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.03em;
    line-height: 1.2;
}

.hero-section .hero-subtitle {
    color: rgba(186, 210, 230, 0.9);
    font-size: 1rem;
    line-height: 1.7;
    max-width: 700px;
    margin: 0 auto;
}

/* ---- Input & Output Panels (consistent 16px radius) ---- */
.input-panel {
    background: rgba(12, 25, 46, 0.25) !important;
    border: 1px solid rgba(56, 189, 248, 0.1) !important;
    border-left: 3px solid rgba(56, 189, 248, 0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

.output-panel {
    background: rgba(12, 25, 46, 0.3) !important;
    border: 1px solid rgba(14, 165, 233, 0.1) !important;
    border-left: 3px solid rgba(14, 165, 233, 0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

/* ---- Section Titles ---- */
.section-title {
    margin-bottom: 0.8rem !important;
}

.section-title h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    color: rgba(226, 232, 240, 0.95) !important;
}

/* ---- Step Buttons (consistent 16px radius, tactile active state) ---- */
.btn-scan {
    border-radius: 16px !important;
    font-weight: 600 !important;
    padding: 13px 26px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
    background: transparent !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    color: #7dd3fc !important;
}

.btn-scan:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.12) !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border-color: rgba(56, 189, 248, 0.55) !important;
}

.btn-scan:active {
    transform: translateY(0px) scale(0.98) !important;
    transition-duration: 0.08s !important;
}

.btn-generate {
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    padding: 14px 28px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    background: transparent !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    margin-top: 0.3rem !important;
    color: #7dd3fc !important;
}

.btn-generate:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.12) !important;
    background: rgba(56, 189, 248, 0.08) !important;
    border-color: rgba(56, 189, 248, 0.55) !important;
}

.btn-generate:active {
    transform: translateY(0px) scale(0.98) !important;
    transition-duration: 0.08s !important;
}

/* ---- Result Tabs ---- */
.result-tabs .tab-nav {
    background: rgba(30, 41, 59, 0.5) !important;
    border-radius: 16px !important;
    padding: 4px !important;
    border: 1px solid rgba(56, 189, 248, 0.08) !important;
    margin-bottom: 0.5rem !important;
}

.result-tabs .tab-nav button {
    border-radius: 12px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
}

.result-tabs .tab-nav button.selected {
    background: rgba(56, 189, 248, 0.15) !important;
    font-weight: 600 !important;
}

/* ---- Status Line ---- */
.status-line {
    min-height: 1.2rem !important;
    margin-bottom: 0.4rem !important;
}

.status-line p {
    color: rgba(125, 211, 252, 0.85) !important;
    font-size: 0.9rem !important;
}

/* ---- Score Badge ---- */
.score-badge {
    text-align: center !important;
    min-width: 180px !important;
    background: rgba(56, 189, 248, 0.06) !important;
    border: 1px solid rgba(56, 189, 248, 0.12) !important;
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.score-badge .prose {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
}

.score-badge .prose h3 {
    margin: 0 !important;
    font-size: 1.3rem !important;
}

/* ---- Download Area ---- */
.download-area {
    background: rgba(30, 41, 59, 0.2) !important;
    border: 1px dashed rgba(100, 116, 139, 0.2) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    margin-top: 0.75rem !important;
    transition: all 0.2s ease !important;
}

.download-area.has-files {
    background: linear-gradient(135deg,
        rgba(34, 197, 94, 0.03),
        rgba(16, 185, 129, 0.03)) !important;
    border: 1px solid rgba(34, 197, 94, 0.15) !important;
}

/* ---- Result Tabs: keep content fully opaque (Gradio dims inactive panels) ---- */
.result-tabs,
.result-tabs *,
.result-tabs .prose,
.result-tabs .prose *,
.result-tabs:hover,
.result-tabs *:hover {
    opacity: 1 !important;
    transition: none !important;
    filter: none !important;
}

/* ---- Divider ---- */
.section-divider {
    opacity: 0.4;
    margin: 0.8rem 0 !important;
}

/* ---- Disabled button state (during pipeline runs) ---- */
.btn-scan:disabled,
.btn-generate:disabled,
button[disabled] {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid rgba(100, 116, 139, 0.15);
    margin-top: 2.5rem;
}

.footer p {
    color: rgba(148, 163, 184, 0.65) !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
}

/* ---- Focus States ---- */
textarea:focus, input[type="text"]:focus {
    border-color: rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.06) !important;
    transition: all 0.2s ease !important;
}

button {
    transition: all 0.2s ease !important;
}

.tabitem {
    opacity: 1 !important;
}

/* ---- Result Textboxes ---- */
.output-panel textarea {
    background: rgba(12, 25, 46, 0.4) !important;
    border: 1px solid rgba(100, 116, 139, 0.12) !important;
    border-radius: 12px !important;
}

.output-panel textarea:hover {
    border-color: rgba(56, 189, 248, 0.18) !important;
}
"""

# ============================================================
# CUSTOM THEME
# ============================================================
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _score_emoji(score_val: int) -> str:
    """Color-coded indicator aligned with the pipeline gate (pass mark = 80)."""
    if score_val >= 80:
        return "🟢"
    elif score_val >= 60:
        return "🟡"
    return "🔴"


# ============================================================
# STEP FUNCTIONS (with progress tracking)
# ============================================================

def _initial_inputs(job_input: str, resume_source: str) -> dict:
    return {
        "job_description": job_input.strip(),
        "original_resume": resume_source,
        "human_notes": "", "missing_skills": "",
        "resume_text": "", "job_text": "", "optimized_resume": "",
        "feedback": "", "score": 0, "iteration": 0,
        "llm_quality_score": 0, "ats_percentage": 0.0, "review_failed": False,
        "cover_letter": "", "interview_questions": "", "ats_result": "",
        "resume_pdf_path": "", "cover_letter_pdf_path": "",
        "resume_docx_path": "", "resume_tex_path": ""
    }


def step1_analyze(job_input, resume_file, resume_text, session):
    """
    Step 1: runs the interactive graph (loader → scanner) which then INTERRUPTS
    before the Improver — a first-class LangGraph human-in-the-loop checkpoint.
    Streams per-node status updates; the thread id is kept in gr.State so
    Step 2 resumes the exact same session (documents are loaded only once).
    Outputs: [btn1, btn2, session, notes, status]
    """
    # --- Input Validation ---
    if not job_input or not job_input.strip():
        raise gr.Error("Please provide a Job Description (text or URL).")

    resume_source = _resolve_resume_input(resume_file, resume_text)
    if not resume_source:
        raise gr.Error("Please upload a resume PDF or paste your resume text.")

    session = session or str(uuid4())
    config = _thread_config(session)

    yield (
        gr.update(interactive=False),  # disable Step 1 while running
        gr.update(interactive=False),  # disable Step 2 (prevents double pipeline runs)
        session, "",
        "Loading documents..."
    )

    try:
        for update in agent_app.stream(_initial_inputs(job_input, resume_source),
                                       config=config, stream_mode="updates",
                                       subgraphs=False):
            for node, delta in update.items():
                if node and delta:
                    yield (gr.update(), gr.update(), gr.update(), gr.update(),
                           f"{NODE_STATUS.get(node, node + '...')} (step 1 of 2)")
        snapshot = agent_app.get_state(config)
        missing = (snapshot.values or {}).get("missing_skills", "")
    except Exception as e:
        yield (gr.update(interactive=True), gr.update(interactive=True),
               session, "", "")
        raise gr.Error(f"Scan failed: {str(e)}")

    gr.Info("Gap analysis complete! Review the results below, then click Step 2.")
    yield (
        gr.update(interactive=True),
        gr.update(interactive=True),
        session,
        missing,
        "Analysis complete — review & edit the gaps below, then run Step 2."
    )


def step2_optimize(job_input, resume_file, resume_text, user_notes, session):
    """
    Step 2: resumes the interrupted session from the checkpoint. If Step 1 was
    skipped, the graph first runs loader → scanner up to the interrupt, then
    the user's notes are injected via update_state and the pipeline continues:
    improver ⇄ reviewer → ats_check → fan-out → pdf_exporter.
    Streams per-node status updates throughout.
    Outputs: [btn1, btn2, session, resume, score, feedback, cover, ats,
              interview, pdf_resume, pdf_cover, status]
    """
    # --- Input Validation ---
    if not job_input or not job_input.strip():
        raise gr.Error("Please provide a Job Description (text or URL).")

    resume_source = _resolve_resume_input(resume_file, resume_text)
    if not resume_source:
        raise gr.Error("Please upload a resume PDF or paste your resume text.")

    if not user_notes or not user_notes.strip():
        gr.Warning("No missing skills / context provided. Running with auto-detected gaps only.")

    def keep_results():
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update())

    session = session or str(uuid4())
    config = _thread_config(session)

    yield (
        gr.update(interactive=False), gr.update(interactive=False),
        session, *keep_results(),
        "Starting pipeline..."
    )

    try:
        state = agent_app.get_state(config)
        if not state.next:
            if state.values:
                # previous run completed in this session → start a fresh one
                session = str(uuid4())
                config = _thread_config(session)
                yield (gr.update(), gr.update(), session, *keep_results(),
                       "Starting a fresh run...")
            # Step 1 was skipped: run loader → scanner up to the interrupt
            for update in agent_app.stream(_initial_inputs(job_input, resume_source),
                                           config=config, stream_mode="updates"):
                for node, delta in update.items():
                    if node and delta:
                        yield (gr.update(), gr.update(), gr.update(), *keep_results(),
                               f"{NODE_STATUS.get(node, node + '...')} (1/2)")

        # Inject the user's edited gaps/context at the human-review checkpoint
        agent_app.update_state(config, {
            "human_notes": user_notes or "",
            "missing_skills": user_notes or "",
        })

        # Resume: improver ⇄ reviewer → ats_check → fan-out → pdf_exporter
        final_state = {}
        for update in agent_app.stream(None, config=config, stream_mode="updates"):
            for node, delta in update.items():
                if node and delta:
                    final_state.update(delta)
                    yield (gr.update(), gr.update(), gr.update(), *keep_results(),
                           f"{NODE_STATUS.get(node, node + '...')} (2/2)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (gr.update(interactive=True), gr.update(interactive=True),
               session, *keep_results(), "")
        raise gr.Error(f"Optimization failed: {str(e)}")

    if not final_state:
        snapshot = agent_app.get_state(config)
        final_state = snapshot.values or {}

    # Extract Results — use empty strings as safe fallbacks
    opt_resume_text = final_state.get('optimized_resume') or "No resume generated."
    score_val = final_state.get('score', 0)
    score_emoji = _score_emoji(score_val)
    score_text = f"### {score_emoji} {score_val}/100\n*composite*"
    feedback_text = final_state.get('feedback') or "No feedback provided."
    cover_letter_text = final_state.get('cover_letter') or "No cover letter generated."
    ats_text = final_state.get('ats_result') or "ATS analysis not available."
    interview_text = final_state.get('interview_questions') or "No interview questions generated."

    # Get PDF/DOCX/TeX Paths — ensure they exist, otherwise pass None
    resume_pdf = final_state.get('resume_pdf_path') or None
    cover_pdf = final_state.get('cover_letter_pdf_path') or None
    resume_docx = final_state.get('resume_docx_path') or None
    resume_tex = final_state.get('resume_tex_path') or None
    if resume_pdf and not os.path.exists(resume_pdf):
        print(f"[Warning] Resume PDF path does not exist: {resume_pdf}")
        resume_pdf = None
    if cover_pdf and not os.path.exists(cover_pdf):
        print(f"[Warning] Cover letter PDF path does not exist: {cover_pdf}")
        cover_pdf = None
    if resume_docx and not os.path.exists(resume_docx):
        print(f"[Warning] Resume DOCX path does not exist: {resume_docx}")
        resume_docx = None
    if resume_tex and not os.path.exists(resume_tex):
        print(f"[Warning] Resume TEX path does not exist: {resume_tex}")
        resume_tex = None

    ats_pct = final_state.get('ats_percentage', 0)
    gr.Info(f"Optimization complete! Final score: {score_val}/100")

    status = (f"Done — composite score **{score_val}/100** "
              f"(ATS {ats_pct:.0f}% x 0.6 + LLM quality x 0.4).")

    yield (
        gr.update(interactive=True),   # re-enable Step 1
        gr.update(interactive=True),   # re-enable Step 2
        session,
        opt_resume_text, score_text, feedback_text,
        cover_letter_text, ats_text, interview_text,
        resume_pdf, cover_pdf, resume_docx, resume_tex,
        status
    )


def _resolve_resume_input(resume_file, resume_text):
    """Resolves which resume input to use: file takes priority over text."""
    if resume_file:
        return resume_file
    if resume_text and resume_text.strip():
        return resume_text.strip()
    return None


# ============================================================
# BUILD THE USER INTERFACE
# ============================================================

with gr.Blocks(title="AI Resume Optimizer Agent") as demo:

    # ───────────── Hero Header ─────────────
    with gr.Column(elem_classes=["hero-section"]):
        gr.HTML(
            '<h1 class="hero-title">AI Resume Optimizer Agent</h1>'
            '<p class="hero-subtitle">'
            'An intelligent agentic workflow that <strong>scans</strong> for gaps → '
            'lets you <strong>review &amp; edit</strong> → then <strong>generates</strong> '
            'optimized documents.'
            '</p>'
        )

    with gr.Row(equal_height=False):

        # ═══════════ LEFT COLUMN — Inputs ═══════════
        with gr.Column(scale=1, elem_classes=["input-panel"]):
            with gr.Column(elem_classes=["section-title"]):
                gr.Markdown("### 📋 Input")

            job_in = gr.Textbox(
                label="Job Description",
                placeholder="Paste the full job description text or a URL (https://...)",
                lines=4,
                autofocus=True,
                info="Paste the complete job posting for best results. URL import only works for allowlisted domains."
            )

            gr.Markdown("**Resume** — choose one method:")
            with gr.Tabs():
                with gr.Tab("📎 Upload PDF"):
                    resume_file_in = gr.File(
                        label="Upload Resume (PDF)",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                with gr.Tab("📝 Paste Text"):
                    resume_text_in = gr.Textbox(
                        label="Paste Resume Text",
                        placeholder="Paste your resume content here...",
                        lines=5
                    )

            with gr.Column(elem_classes=["section-divider"]):
                gr.Markdown("---")

            btn_analyze = gr.Button(
                "🔍 Step 1 — Analyze Skill Gaps",
                variant="secondary",
                elem_classes=["btn-scan"]
            )

            notes_in = gr.Textbox(
                label="✏️ Review & Edit — Missing Skills",
                placeholder=(
                    "Click 'Step 1' above to auto-detect missing skills.\n"
                    "Then edit: remove skills you lack, add context for ones you have.\n"
                    "Example: 'I used SQL during my internship at XYZ...'"
                ),
                lines=10,
                interactive=True,
                info="Your edits here directly guide the resume optimizer"
            )

            btn_optimize = gr.Button(
                "Step 2 — Generate All Documents",
                variant="primary",
                elem_classes=["btn-generate"],
                interactive=False
            )

        # ═══════════ RIGHT COLUMN — Results ═══════════
        with gr.Column(scale=1, elem_classes=["output-panel"]):
            with gr.Column(elem_classes=["section-title"]):
                gr.Markdown("### 📊 Results")

            status_out = gr.Markdown(
                value="",
                elem_classes=["status-line"]
            )

            with gr.Row():
                score_out = gr.Markdown(
                    value="### ⏳ Awaiting results...",
                    elem_classes=["score-badge"]
                )
                feedback_out = gr.Textbox(
                    label="Reviewer Feedback",
                    lines=3,
                    interactive=False
                )

            with gr.Tabs(elem_classes=["result-tabs"]):
                with gr.Tab("📄 Resume"):
                    resume_out = gr.Markdown(
                        value="*Run **Step 2** to see your optimized resume.*",
                        label="Optimized Resume"
                    )

                with gr.Tab("✉️ Cover Letter"):
                    cover_letter_out = gr.Markdown(
                        value="*Run **Step 2** to see your cover letter.*",
                        label="Cover Letter"
                    )

                with gr.Tab("📊 ATS Match"):
                    ats_out = gr.Markdown(
                        value=(
                            "#### ⏳ Waiting for results...\n\n"
                            "*Run **Step 2** to see your ATS keyword analysis.*"
                        ),
                        label="ATS Keyword Match"
                    )

                with gr.Tab("🎤 Interview Prep"):
                    interview_out = gr.Markdown(
                        value=(
                            "#### ⏳ Waiting for results...\n\n"
                            "*Run **Step 2** to generate targeted interview questions.*"
                        ),
                        label="Interview Preparation"
                    )

            # PDF/DOCX Downloads
            with gr.Column(elem_classes=["download-area"]):
                gr.Markdown("#### 📥 Download Documents")
                with gr.Row():
                    pdf_resume_out = gr.File(label="Resume PDF")
                    pdf_cover_out = gr.File(label="Cover Letter PDF")
                with gr.Row():
                    docx_resume_out = gr.File(label="Resume DOCX (Word)")
                    tex_resume_out = gr.File(label="Resume TEX (LaTeX source)")

    # ───────────── Footer ─────────────
    with gr.Column(elem_classes=["footer"]):
        gr.Markdown(
            "Built with ❤️ using **LangGraph** · **Groq (GPT-OSS 120B)** · **Gradio** · "
            "**LangSmith**  \n"
            "© 2025–2026 AI Resume Optimizer Agent"
        )

    # ═══════════ WIRING ═══════════

    # Per-browser-session thread id — keeps the LangGraph checkpoint (and the
    # loaded documents) tied to this user's two-step flow
    session_state = gr.State(None)

    btn_analyze.click(
        fn=step1_analyze,
        inputs=[job_in, resume_file_in, resume_text_in, session_state],
        outputs=[btn_analyze, btn_optimize, session_state, notes_in, status_out],
        show_progress="hidden"
    )

    btn_optimize.click(
        fn=step2_optimize,
        inputs=[job_in, resume_file_in, resume_text_in, notes_in, session_state],
        outputs=[
            btn_analyze, btn_optimize, session_state,
            resume_out, score_out, feedback_out,
            cover_letter_out, ats_out, interview_out,
            pdf_resume_out, pdf_cover_out, docx_resume_out, tex_resume_out,
            status_out
        ],
        show_progress="hidden"
    )

if __name__ == "__main__":
    init_output_dirs()

    # One pipeline at a time: concurrent runs would double-spend Groq quota
    demo.queue(max_size=5, default_concurrency_limit=1)

    # Optional basic auth for public deployments — set both env vars to enable.
    # Without it, anyone who can reach the app can spend your Groq quota.
    auth_user = os.environ.get("GRADIO_AUTH_USERNAME")
    auth_pass = os.environ.get("GRADIO_AUTH_PASSWORD")
    auth = (auth_user, auth_pass) if auth_user and auth_pass else None
    if auth is None:
        print("[Auth] No GRADIO_AUTH_USERNAME/GRADIO_AUTH_PASSWORD set — app is unauthenticated (fine for localhost).")

    # Gradio 6.x: theme/css are launch() parameters (moved back from Blocks constructor)
    demo.launch(theme=THEME, css=CUSTOM_CSS, auth=auth)