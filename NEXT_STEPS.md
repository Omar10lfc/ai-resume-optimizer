# Next Steps — Job Application Optimizer

Last updated: **Oct 21 2026** · Snapshot: 131 tests passing (~27s) · coverage 87% (CI gate 84%) · `pip check` clean · langgraph stack 1.2.16 · working tree clean

This is the living roadmap. The public, permanent record of *what was done and why* lives in [README.md](README.md) ("Changelog & Design Decisions" → "Session — Oct 21 2026"); this file tracks **what's next** and the current state. Legend: ✅ done this cycle · 🔲 planned · 🔶 optional.

---

## Phase 1 — Land current work & hygiene ✅ COMPLETE

| # | Item | Outcome | Commit |
|---|------|---------|--------|
| 1 | Land uncommitted fix set | Literal `\n`-escape normalization (loader + ATS) + LaTeX heading detection + 2 regression tests, verified on 112 green tests first | `540bc0b` |
| 2 | README corrections | Test counts 108 → 112 (now 131), CI "110-test suite" fixed, NODE 8 → 7 renumber | `539e618` |
| 3 | SQLite checkpointer dependency | `langgraph-checkpoint-sqlite` added; **cross-process persistence PASS** proved on live code (2 processes, shared DB) | `3b00a56` |
| 4 | Tree cleanup | Stale root `__pycache__` (67 KB, incl. deleted `resume_agent.py` monolith bytecode) removed; `scratch/` gitignored | `3e6f114` |
| 5 | Full E2E smoke run on live Groq | Bug-window condition reproduced (literal-escaped JD + real CV): **PASS** — extraction clean, composite 67/100, all 4 PDF/DOCX/TeX artifacts generated; app-faithful traced re-runs exported healthy baseline **76/100 (ATS 72%, LLM 82)** + reviewer-failure evidence run | traces/ |

## Phase 2 — Short-term hardening ✅ COMPLETE

| # | Item | Outcome | Commit |
|---|------|---------|--------|
| 5 | Golden ATS fixtures | `tests/fixtures/golden_ats.json` (real incident JD + synthetic resume) + 5 tests: escaped/real extraction identical, skills survive, garbage filtered, 30% weighted floor | `09b5aa1` |
| 6 | CI improvements | Matrix **3.11 + 3.12**, `pytest-cov` gate `--cov-fail-under=84` (measured 87%), dev deps split to `requirements-dev.txt` | `30d74f9` |
| 7 | xhtml2pdf decision | **Decision: KEPT** (documented in README) — pure-Python, HTML-sanitized, cover letter is simple prose; revisit triggers recorded | — |
| 8 | DOCX resume ingestion | `.docx` end-to-end via `python-docx` (paragraphs + tables, 5 MB cap, magic-byte check); Gradio accepts PDF/DOCX; graph-level test | `30d74f9` |
| 9 | Reviewer structured-output retry | One same-payload retry before `review_failed`; trace-evidenced (~1-in-3 transient folds); give-up-after-2 tested | `30d74f9` |
| 10 | Startup provider health check | One `GET /models` call (zero generation tokens) with actionable verifications (missing/placeholder key, 401/403, 429, unreachable, primary-model-not-available); `SKIP_STARTUP_HEALTHCHECK=1` opt-out; 8 branch tests | `30d74f9` |

**(Bonus):** langgraph stack upgraded **1.0.5 → 1.2.16** (checkpoint 4.x pairing, closes the version-skew trap; `pip check` clean, persistence + live smoke revalidated, band unchanged 67–76) — `bd713a1`.

---

## Phase 3 — Feature roadmap 🔲 OPEN (priority order)

### 11 🔲 Before/After diff view — first candidate
Side-by-side original vs optimized resume in Gradio with changed bullets/skills highlighted and a score-delta badge (before/after ATS % + composite).
- *Implementation sketch:* streamer re-emits `original_resume` alongside `optimized_resume`; diff via `difflib.SequenceMatcher` (word-level) + `rich`-style markup or two Gradio `HTML` panes; no new LLM calls, no new state fields beyond wiring.
- *Effort:* M (~1 session). *Why it leads:* highest user-perceived value; directly demonstrates the product's promise.

### 12 🔲 Google XYZ bullet impact scorer
Audit every bullet against "Accomplished [X] as measured by [Y] by doing [Z]"; flag bullets with no metric/weak verb.
- *Implementation sketch:* deterministic first pass (regex for numbers/percent/'$'/strong verbs), optional LLM pass with structured output (reuse `SemanticReview` pattern); new results tab "Bullet Impact".
- *Effort:* M. *Care:* add retry like the reviewer once trace data exists.

### 13 🔲 Multi-provider LLM support (Groq + OpenAI + Anthropic + local Ollama)
Refactor `agent/llms.py` around `langchain.chat_models.init_chat_model` behind an `LLM_PROVIDER` env var; keep the 3-profile structure (strict/creative/fast) and the fallback chain semantics.
- *Care:* `reasoning_effort` is Groq/GPT-OSS-specific — gate per provider; health check needs provider-specific endpoints (current `_groq_models_status` is Groq-only).
- *Effort:* M-L (~1–2 sessions). *Value:* removes single-vendor dependency; notable demo value.

### 14 🔶 Multi-language support
Snowball stemmers (~15 languages) per the known non-Latin keyword limitation + `PROMPT_LANGUAGE` param for model-facing prompts.
- *Only if needed* for actual usage; effort M; pairs with the backlog note on ASCII-only extraction.

### 15 🔶 HF Spaces deploy polish
Pin the Space's `gradio` version to match, ensure 7860 binding default, `.gitattributes` LFS only if PDFs become assets; verify `SKIP_STARTUP_HEALTHCHECK` behavior in Space startup logs.

---

## Phase 4 — Larger bets 🔶 OPTIONAL

| Bet | Notes |
|-----|-------|
| LangSmith eval set | Replay exported traces + fresh runs into a dataset (score sanity ranges, no fabricated sections, keyword precision). **Must use `client.runs.query()`** — `list_runs()` deprecates after **Jan 31 2027** |
| Dockerfile | Bake LaTeX toolchain for reproducible deploys beside HF Spaces path |
| Frontend rebuild (Streamlit / FastAPI+React) | Only if Gradio blocks something concrete; current UI is tested and streaming-capable |

---

## Backlog candidates found this session

1. **Semantic-verify retry** (`ats_check_node::_semantic_verify_missing`) — same structured-output transient-failure family as the reviewer; currently fails soft (keeps keywords as missing). Consider the same retry-once pattern once traces show its failure rate.
2. **Top-20 frequency bias** — on the incident JD, `python`/`git`/`llm` lost the top-20 cut to generic words (`workflows`, `ai`); smoke preflight flagged it. Not a defect; a ranking-quality observation for the ATS extractor.
3. **Ubuntu vs Windows coverage drift** — the 84% gate has headroom (measured 87% on Windows); if CI (Linux) trends near the gate, consider measuring coverage locally with the same flags before raising/flooring it.

## Verification playbook

```bash
# Full suite (no network / no API cost)
venv\Scripts\python.exe -m pytest tests -q

# Coverage check (mirrors CI gate)
venv\Scripts\python.exe -m pytest tests -q --cov=agent --cov=app --cov-report=term --cov-fail-under=84

# Live traced smoke run (one real pipeline run; exports trace to traces/)
# script kept in %TEMP%\opencode\smoke_run_traced.py during the session

# SQLite checkpointer persistence (two fresh processes)
# scripts kept in %TEMP%\opencode\ckpt_put.py / ckpt_get.py
```

**Definition of done for any new phase item:** feature code + tests green in the full suite, coverage gate still passing, live smoke run re-verified when the pipeline's LLM path changes, README changelog row (change/why/impact) added, this file's status updated.
