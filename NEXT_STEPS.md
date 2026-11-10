# Next Steps — Job Application Optimizer

Last updated: **Nov 10 2026** · Snapshot: 252 tests passing (~2:15) · coverage 85% (CI gate 84% over agent+app+server) · langgraph stack 1.2.16 · FastAPI `server/` + Vite React `frontend/` shipped & live-probed · **Chrome extension MVP + UI redesign committed · Nov 8: `539f7c1d` stale-chip laundering closed + PDF entry-boundary fix · Nov 10: Dossier-study round landed (QW1 prompt stamps / QW2 decline protocol / QW3 integrity UI / MR1 section-targeted polish), suite 252**

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

## Skill-pack behavior alignment ✅ (Oct 21, part 2)

The seven professional skill definitions kept locally in `draft/` (gitignored) were mined for behavioral principles and encoded across the agent — details and per-node mapping live in README changelog → "Prompt-behavior alignment with the skill pack" (`5359422`, `546397a`). Summary: scanner gaps are ranked and must/nice-tagged with hard-skill priority; the improver applies the tailoring philosophy, exact-role summaries, relevant-first skill ordering, and X-Y-Z bullets with a deterministic weak-bullet detector feeding rewrite targets; the cover letter uses hook openings + call-to-action closings with banned clichés; interview prep adds a STAR story bank, pitch, and questions-to-ask; the reviewer rubric is anchored to those standards (first aligned live run: LLM quality 85 vs 82, composite 72/100 — band holds). Suite **131 → 136 tests**.

**Partial revert (`30b2e23`, user decision):** the resume-**rewriting** path (scanner + improver + weak-bullet detector) was reverted to its prior, live-validated behavior — the output-stage alignment (cover letter, interview prep, reviewer rubric) remains in place. Suite back to **131 tests**.

---

## Product layer — FastAPI server + React frontend ✅ SHIPPED (Oct 24–28)

The Phase-4 "frontend rebuild" trigger fired. Details in README changelog → "Session — Oct 24–28 2026".

| # | Item | Outcome | Commit |
|---|------|---------|--------|
| P1 | FastAPI API server | `server/`: SSE step1/step2 + health + artifact downloads + optional auth + Gradio-equivalent queue semantics; 23-test suite | `4ffb4fe`, `73c4cf4` |
| P2 | Vite React frontend | Two-step wizard, gap-review step, node-driven streaming timeline, results tabs + download center; `tsc && vite build` clean | `0dd08bb`, `5bf3565` |
| P3 | Contract hardening | Status events carry graph node ids (fan-out + correction loop display correctly); `warning`/`session` SSE events parsed; reviewer-feedback panel restored; weights rendered from server payload | `5bf3565` |
| P4 | Live validation | Real-stack probe (fake candidate, no real PII): step1 wire exact, step2 scored 70/100 with all four artifacts downloaded live; queue/auth/validation paths probed zero-cost through the dev proxy | — |

**Remaining frontend polish:** ✅ light-mode + token theming shipped (Nov 3 redesign, shared with the extension); still open: dynamic-import the markdown tabs (570 kB chunk warning), single-container production deploy superseding the Gradio Space.

---

## Browser extension — Chrome MV3 side panel 🔨 MVP BUILT, UNCOMMITTED (Nov 2)

Third API client per the product implementation plan; reuses the same FastAPI API (no separate stack). Build: `cd frontend && npm run build:extension`, then "Load unpacked" on `extension/` — details in [extension/README.md](extension/README.md).

| # | Item | Outcome |
|---|------|---------|
| X1 | MV3 scaffold + side panel | Manifest (`sidePanel`/`storage`/`activeTab`/`scripting`, Chrome ≥ 114), React app compiled into the panel, ESM background worker, floating-button content script (IIFE); toolbar click + button click both open the panel |
| X2 | JD scraper | Site registry (LinkedIn, Greenhouse, Lever, Indeed, Workday, Ashby) + generic fallback; scraped payload lands in the panel's job form with a title/company badge |
| X3 | **Server BYO-key** | The `X-Groq-Api-Key` header (sent by the existing settings UI) is now honored: per-request key override in `agent/llms.py` (key-scoped chains + restore-on-exit), threaded via `server/app.py` → `runners`; `/health` validates the user key; keys capped/sanitized; 5 new tests |
| X4 | Resume vault | Resume text persisted & restored across sessions (chrome.storage / localStorage); stale saved text purged when the user clears it; artifact downloads resolve against the configured backend URL |
| X5 | Hygiene | `typecheck:extension` script + strict tsconfig for `extension/src`; stale mirrored assets cleaned on rebuild; `.env.example` documents `APP_AUTH_*`, `EXTENSION_ORIGINS`, BYOK |
| X6 | **UI/UX redesign (Nov 3)** | Design-token CSS system with light+dark themes (system-follow + manual toggle, `uiTheme` storage key); gap chips grouped tool/concept/general with per-chip notes/undo/keyboard map; classified error card with actions; score gauge with ATS/quality split; elapsed-timer timeline with staleness hints; before/after diff in paste mode; ARIA tabs; debounced storage writes; 3-step onboarding; "forget my data" eraser — see README changelog "Session — Nov 3 2026" |
| X7 | **Gap-flow behavior (Nov 5)** | Review→step-2 handoff race fixed (`onSubmit(notes: string)` passes the composed string — state reads in the same tick were stale); zero-gap scans skip the review modal and optimize straight away; new optional "Add context for the optimizer" box under the inputs persists across sessions (`pendingUserNotes` / `pending_optimizer_notes`, purged by Reset and the Settings eraser); timeline gains the polisher pass-2 row — see README changelog "Session — Nov 5 2026" |

**Next for the extension:** CWS store assets (screenshots, promo tile, privacy policy), scraper fixture tests, permission review (broad `https://*/*` host permission needed only for a configurable cloud backend — consider `optional_host_permissions`), a live LinkedIn smoke pass, and a reload of the unpacked extension in `chrome://extensions` to pick up the Nov 5–7 panel changes (bundle `index-ejUvToFN.js` — the Nov 6–7 fabrication incident traced back to a stale panel bundle, so this reload matters).

---

## Agent core — Nov 5 2026 (straight-through tailoring, ATS round two, best-of-2 loop) ✅ LANDED (`10d48ca` + panel-side commit)

Motivated by the user's live run (trace `520188c7`, request `3ffdcffc`): the human-review handoff race dropped composed gap notes, the ATS scored posting boilerplate, and the reviewer's feedback never reached the improver because `MAX_ITERATIONS=1` stopped the loop at the cap. Panel fixes (App/Modal/Settings/Timeline) ship in this session's frontend/doc commit.

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A1 | Gap handoff fix | `onSubmit(notes: string)` carries the composed review directly; retries/auto-run fall back to `userNotesRef` | Race: React state committed too late for the step-2 request |
| A2 | Zero-gap straight-through | `reviewOpen` gating + toast + auto-run when the scan finds no gaps | The JD-tailoring flow is automatic; review = context-adding only |
| A3 | Optimizer-context box | Persistent optional notes under the inputs (extension + web keys), eraser-aware | Also pre-fills the gap editor when it opens |
| A4 | ATS hygiene round two | Perk/EEO/apply-section stripping + timeline/logistics/qualifier stopwords + camelCase brand filter | Probe: noise zero, core tech kept, 73.3% synthetic ATS |
| A5 | `devsecops` regression fixed | `_CAMEL_TECH_ALLOW` extended; regression test added | Caught by the golden fixture during validation |
| A6 | Best-of-2 loop | `MAX_ITERATIONS` 1→2 + `best_result` snapshot restore in `ats_check`; polisher node routing (interrupt stays on first improver visit) | Extra polish can never regress the final result |
| A7 | Suite 191 → 199 | 8 new agent tests; salvage test pinned to the loop cap for exact retry counting | Full-suite green before commit |

---

## Agent core + panel — Nov 6–7 2026 (fabrication guardrails, pass-case polish, gauge contrast) ✅ LANDED (this session's two commits)

Driven by the user's live run (request `14751ff9…`, UI session `8dcb2cab`, LangSmith roots `01a1b28e` + `c71e0454`): the run was healthy (86 composite / 90.5% ATS / 1 iteration / 1-page PDF) but the optimized resume + cover letter **fabricated tools and a metric** (n8n "cut … by 40%", FortyGuard MCP integration, Claude Code/Cursor/Codex) from gap chips with EMPTY user notes — the stale panel bundle kept re-dropping the composed review, and the improver prompt's "plausibly has" line licensed the invention. The screenshot's "broken" quality bar turned out to be a contrast bug (`--line-strong` border token on a dark inset track).

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A8 | Integrity tripwire | `find_unevidenced_metric_claims` (spacing-compact matching; years/versions exempt) → improver stamps `guardrail_warnings` → runner emits a `warning` SSE event before `result` | Detection-only; run 14751ff9's invented "40%" would now be surfaced to the user |
| A9 | Reviewer truthfulness gate | Score capped at 40 + invented claims named in feedback; ORIGINAL RESUME embedded as the evidence anchor | Fabrication is graded, not just prompted against |
| A10 | Improver prompt rewrite | Gap items = RESEARCH TOPICS; named tools/integrations only when evidenced; zero-tolerance metric rule | "Plausibly has relevant experience" line removed |
| A11 | Pass-case polish | `should_continue` routes to `polisher` on non-empty feedback whenever the cap allows (cap first; `review_failed` fans out) | Run 14751ff9's "adjust the professional summary" suggestion is now applied on the next pass; best-of guard prevents regressions |
| A12 | Bullet Unicode hygiene | `_tame_unicode_bullets` on scanner output + gap-chip labels (`Claude\u202fCode`, `Hands\u2011on`) | Chips/ATS spans get plain ASCII separators |
| A13 | Zero-gap gate hardened | Auto-run only when `gapCount === 0 && !scannerOutput` | Unparseable-but-nonempty scanner text opens the review modal instead of bypassing the decision surface |
| A14 | Score-gauge contrast | ATS bar `var(--accent)`, quality bar `var(--info)`, per-bar color-coded % labels, tightened ring-center type | The 79% quality fill is visible again (was border-token-on-inset) |
| A15 | Suite 199 → 211 (agent 112, server 29) | Pass-case polish ×3, integrity ×5, Unicode ×3, reviewer-gate asserts, server warning-event ordering test | Full suite green; frontend lint/tsc/typecheck/builds clean; bundle `index-Czb7PI05.js` |

**⚠️ Run 14751ff9 artifacts (`Optimized_Resume (10).pdf` etc.) must not be sent out** — they contain the fabricated n8n/FortyGuard-MCP/Claude-Code claims. Regenerate after reloading the unpacked extension in `chrome://extensions`.

---

## Agent core — Nov 7 2026 (run pair: traces + chip truncation + PDF overlap) ✅ LANDED (this session's two commits)

Driven by TWO back-to-back live runs with scraped JD context with/without user notes (LangSmith roots `01a1b310`+`39be45ad` = no-context run, score 67 / ATS 85.7 / quality 40 / iter 2, request `a32f30fc…`; `01a1b317`+`763249a5` = context run, score 61 / ATS 76.2 / quality 38 / iter 2, request `868e406d…`). The traces proved the mechanics healthy (polisher + reviewer gate + tripwire all live) but the fabricated n8n/Claude-Code/MCP bullets STILL shipped, the downloaded run-2 PDF printed header rows onto bullet lines, and tripwires flagged the user's REAL numbers.

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A16 | Stamping split (`server/runners.py`) | `_split_composed_notes`: `missing_skills` = kept-chip lines (+ `Do NOT feature`), `human_notes` = ONLY the typed evidence | Root cause of the fabrications: the composed notes string (chips under pasted project text) was stamped into BOTH fields, so the model folded chips into the Aegis project as user-confirmed bullets |
| A17 | Deterministic scrubber (`agent/helpers.py` `scrub_unsupported_content`, wired in improver **and polisher**) | Gap-echo bullets removed wholesale; skills rows lose unevidenced items + `(or similar)` hedges; prose loses offending sentences; unevidenced metric claims excised (paren → comma segment → whole line) with `_tidy_removal_residue` cleanup | The reviewer's "remove the unverified claims" feedback was ignored TWICE by the model — detection alone cannot clean the artifact; removal notes flow into `guardrail_warnings` → the warning SSE event |
| A18 | Tripwire whitespace fix | Token compaction strips EVERY whitespace class + per-line compact evidence matching | Run 868e406d flagged `42.69 %` (U+202F narrow space in draft) vs source `42.69%` — the user's real WER numbers; after the fix ALL run-pair numeric flags resolve to evidenced copies (only `< 15 ms` traces to the user's own pasted context) |
| A19 | Compact-pass rescale (`latex_render.py`) | Auto-compact `vspace` overrides -9/-9/-9/-8/-3pt → -4/-4/-4/-4/-2pt | Measured run-2 header→bullet baselines had collapsed to 2.5–2.9pt (glyph overprint in the shipped PDF); now 7.0pt min gap, still 1 page |
| A20 | Date/label parsing | `_MONTH_RANGE_RE` (month-to-month `Aug – Sep 2026`), `_STANDALONE_DATE_RE` whole-segment pop for 4-field headers, trailing bare-label de-glue (`Live Demo`/`GitHub` without URLs) | Run-2's `Aug – Sep 2026` was silently dropped from the project header; `Live Demo` was glued onto the Aegis title |
| A21 | Panel chip truncation fix (`HumanReviewModal.tsx`) | Gap chips wrap to a 2-line clamp with `[overflow-wrap:anywhere]` (was single-line `truncate`), full label stays in the tooltip | The truncation visible in the run-review screenshot; extension needs a chrome://extensions reload to pick the rebuilt bundle `index-ejUvToFN.js` |
| A22 | Suite 211 → 229 | Scrubber ×6, tripwire exotic-space ×1, splitter ×4 (server), LaTeX date/label/collision regressions ×4 (incl. pypdf baseline-gap ≥ 6.5pt + 1-page compact on the overflow fixture) | Full suite green; frontend lint/tsc/typecheck/builds clean |

**⚠️ Run pair artifacts (`a32f30fc…` / `868e406d…` directories and `Optimized_Resume (10).pdf`) must not be sent out** — regenerate with a fresh run after reloading the extension (the old panel bundles still stamp the unsplit notes; the new pipeline splits, scrubs, and warns).


---

## Agent core — Nov 7 2026 (second run pair: honest ATS + precision fixes) ✅ LANDED (this session's two commits)

Driven by the next back-to-back pair on the fixed pipeline — same FortyGuard AI Automation Intern JD (826 chars): no-context request `d8756ddf…` (roots `01a1b4cb`+`09cf51fc`, 94.9s, score 52 / ATS 33.3 / quality 80), with-context request `a9da3660…` (roots `01a1b4d1`+`1d3e1550`, 84.7s, score 54 / ATS 38.1 / quality 78). Verdicts: the score drop vs Nov 6 is NOT a regression (67/61 were fabrication-inflated — fabricated n8n/MCP bullets matched the JD's own vocabulary; honest drafts score ATS ~33–38% against an automation JD the ML/CV resume genuinely lacks, quality 80/78); the slower wall-clock is NOT the review interrupt (it adds no LLM time — the ~40s polish call is the Nov 5 best-of-2 loop, and the reviewer structured-output flake `BadRequestError 400 "Tool choice is required"` burned 41.7s/27.4s, salvaged both times by the `failed_generation` prose parser); the screenshot PDF (run B) is clean — 1 page, no overprint, every number verbatim from the user's pasted Aegis context, and both final PDFs contain zero n8n/Claude-Code/MCP (scrubber live on real runs: A removed 1 gap-echo bullet, B removed 5 bits).

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A23 | Scrubber chip-term precision (`agent/helpers.py`) | `_significant_terms` extracts TECHNOLOGY NAMES only: capitalized or digit-bearing tokens; consecutive names in one punctuation segment join into ONE multi-word term (`claude code`, `fortyguard temperature mcp`); lowercase-only chips yield no terms | Run `09cf51fc` culled an innocent draft bullet on the chip-filler term "using" — the old every-non-stopword heuristic turned chip descriptions ('agent', 'development', 'workflows-automation', 'environments') into removal triggers; `_term_in_text` now matches multi-word terms across whitespace runs so `code` can never hit "code coverage" |
| A24 | ATS posting-header + prose-verb stoplist (`agent/ats.py`) | Header fields (`location/months/paid/type/enables` …) and prose verbs (`interact/understand/built/internal/someone/efficiently` …) stopped; `platform`/`servers` low-signal | The tiny scraped JD's plain-text header block put 5 of the 20 weighted keywords into words NO resume can contain, depressing every honest match; the real JD now extracts 19 real keywords leading with `apis/ai/agents/temperature/mcp/claude/cursor/codex` |
| A25 | Reviewer FEEDBACK SCOPE clause (`agent/nodes/reviewer.py`) | Grade PRESENTATION QUALITY only; never recommend adding tools/employers/metrics the evidence lacks | Salvaged reviewer prose in BOTH runs advised "explicitly mention experience with n8n / Claude Code / MCP" — advice only fabrication can follow, feeding the polish loop straight into the scrubber's removals |
| A26 | Tripwire fraction pass-rates (`agent/helpers.py` `_METRIC_TOKEN_RE`) | `N/N` fractions flagged only when followed by benchmark/test/pass/case/check words | `150/150 benchmark passes` is metric-shaped with no %/x/ms unit — could have escaped the unevidenced-metric tripwire if the model coined it; bare "top 5/47" stays exempt, the evidenced "150/150" stays quiet |
| A27 | Suite 229 → 234 | Chip-term shapes ×1, innocent-bullet survival ×1, multi-name true culls ×1, posting-header stoplist ×1 (literal FortyGuard JD fixture), fraction pass-rate ×1, plus anchors-test FEEDBACK SCOPE assertions | Full suite green via `venv\Scripts\python.exe -m pytest tests -q` |

**⚠️ Only the Nov 6 artifacts remain hazardous** (`a32f30fc…` / `868e406d…` directories + `Optimized_Resume (10).pdf`) — the fabricated-bullet residue. This pair's artifacts (`d8756ddf…` / `a9da3660…`) are truthful-by-construction once the user re-verifies their own pasted claims. The new agent code needs a **backend server restart** to load; the extension bundle is unchanged this round (no `chrome://extensions` reload needed).


---

## Agent core + PDF pipeline — Nov 8 2026 (run `539f7c1d`: stale-chip laundering closed + entry-boundary overprint fix) ✅ LANDED (this session's two commits)

Driven by the newest run — request `539f7c1d-9962…` (roots `01a1baec`+`f56c84aa`, 111.9s, **score 75 = ATS 100.0 / quality 38 / iter 2**, same FortyGuard JD, 771 chars). Verdicts: the score is the WORST outcome — the user's pasted step-2 context carried the PREVIOUS session's gap-chip lines verbatim under the Aegis evidence, `_split_composed_notes`'s no-vanish fallback kept them in `human_notes`, the writer shipped them as user-confirmed content (Summary + 3 Skills rows), the scrubber (notes = evidence by contract) found nothing, ATS matched the JD's own vocabulary at 100%, and the reviewer capped quality ~38; controlled replay proves code-vs-inputs: repo scrubber on the run's draft with clean chips → **6 removals**, with the EXACT runtime args (stale chips inside the 802-ch notes) → **0** (inputs poisoned, code fine). The user's "spaces between text" complaint is a second real defect: compact-pass entry boundaries measured **7.0pt** baseline distance (vs 12.0–12.2pt wrapped lines), 10pt descenders printed through bold titles ("Smart Lecture Assistant", "AI Resume Optimizer Agent") — confirmed with a PyMuPDF render. 111.9s is pure LLM latency (no salvage); the `Tool choice is required` 400 flake migrated to the ats_check semantic verifier, caught by its try/except (~2.9s lost).

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A28 | Stale-chip gap-channel hardening (`server/runners.py` `_split_composed_notes`) | Any ASCII-bullet line (`- `/`* `) in composed notes routes to the GAP channel even when it matches no snapshot line; typed evidence keeps `•` bullets/prose | Run `539f7c1d`: stale chips stamped into the evidence channel laundered into user-confirmed fabrications shipped verbatim in Summary + Skills; safe failure direction documented (a typed `- ` note is merely omitted by the writer; classifying a chip as evidence is fabrication laundering) |
| A29 | Scrubber notes-evidence filter (`agent/helpers.py` `scrub_unsupported_content`) | ASCII-bullet lines stripped from `human_notes` before the evidence corpus is built | Belt-and-braces under A28: chips that reach the evidence channel can no longer verify their own echoes — poisoned notes produce exactly the removals clean notes do |
| A30 | PDF entry-boundary repair (`agent/latex_render.py` compact `\resumeItemListEnd` −4pt → +1pt; `agent/templates/resume_template.tex` base −5pt → −1pt) | Compact into-title boundaries 11.9–12.8pt (was 7.0pt), base 18.9–20.8pt; the run's own polished resume still renders 1 page compact; min gap 7.5pt = visually-clean subtitle→bullet pairs | The screenshot complaint: real glyph overprint, not styling preference — the compact end-of-list spacer collapsed bullet→next-title baselines below the 10pt-font descender clearance |
| A31 | Suite 234 → 236 | Splitter: exact `539f7c1d` composition fixture + hardened no-snapshot contract (×3 new/updated); scrubber: notes-poisoning equivalence test (×1); LaTeX: ≥10pt into-title boundary floor in both modes + the overflow fixture rebalanced to moderate + the realistic-dense test now mirrors the exporter's base→compact shipping contract | Full suite green via `venv\Scripts\python.exe -m pytest tests -q` |

**⚠️ Hazardous artifacts now**: the Nov 6 directories (`a32f30fc…` / `868e406d…`) + `Optimized_Resume (10).pdf`, AND this run's directory `%TEMP%\resume_optimizer_output\539f7c1d-9962-41bf-8612-ee3bb8482805\` (PDF/DOCX/TEX/Cover_Letter all carry the fabricated n8n/MCP/Claude-Code Summary + Skills content) — **do not send either**. The Nov 7 pair (`d8756ddf…` / `a9da3660…`) remains truthful. This round's fixes need a **FastAPI backend restart** to load; frontend/extension untouched (bundle stays `index-ejUvToFN.js`).


---

## Agent core + results UI — Nov 10 2026 (Dossier study round: prompt forensics, decline protocol, integrity UI, section-targeted polish) ✅ LANDED
Driven by the repo study of AienTech/dossier — license verdict **AGPL-3.0** (no code reuse; the selected ideas were re-implemented from scratch), and the full ranked comparison lives in **[DOSSIER_COMPARISON.md](DOSSIER_COMPARISON.md)** (repo root). This round ships what the study ranked as QW1/QW2/QW3 + MR1: the "shrink the model's blast radius + verify deterministically" philosophy applied where our scrubber chain already owned the verification half. All four items were chosen by the user (question prompt, Nov 10).

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A32 | Prompt-version stamps, every LLM node (QW1) | New `prompt_versions` state channel; one stamp per node (`scanner@1` / `improver@2` / `polisher-full@1` / `polisher-target@1` / `reviewer@2` / `ats-semantic@1` / `cover-letter@1` / `interview-prep@1`), surfaced in the SSE `result` payload | LangGraph reducer (`_merge_prompt_versions` in `agent/state.py`) merges the channel so the concurrent cover-letter/interview-prep fan-out keeps BOTH stamps — plain last-writer channel would nondeterministically drop one sibling; the server therefore also pulls the merged value from the final graph snapshot (`server/runners.py`), not the in-loop delta accumulation. "Which prompt produced this draft" is now one payload lookup instead of the `539f7c1d`-style trace scavenging |
| A33 | Decline protocol in the writer (QW2) | `improver@2` prompt instruction 8 + `agent/helpers.py` `extract_declined_requests`; new `declined_requests` state channel (accumulated across passes) | Unsupported requests (the n8n/MCP chip class from runs `868e406d`/`539f7c1d`) must be answered with `DECLINED: <request>` lines; parsed out and stripped before ATS/LaTeX/PDF see them, surfaced in the result payload — an omission is now stated by the writer, never fabricated, never silent. The scrubber stays as the deterministic backstop (belt: scrubber; braces: decline) |
| A34 | Integrity view in the results UI (QW3) | `server/runners.py` `_extract_results` adds `removals` / `unevidenced_metrics` / `declined_requests`; `frontend/src/components/ResultsDashboard.tsx` gains a conditional "Integrity" tab (QW3) | Removals become a visible "what we removed & why" list instead of a one-shot SSE warning wall of text; declined requests render with a "the writer declined" explainer; unevidenced metrics render as verify-chips; the tab plus the run's prompt-version line close the forensic loop for non-technical users. Extension bundle + SPA rebuilt (`index-QIO8twRX.js`, `npm run build:extension`) since they share the source |
| A35 | Section-targeted polish (MR1, biggest win) | New `agent/nodes/polisher.py`; reviewer `ReviewOutput.sections` (fixed vocabulary) + prompt block; `agent/helpers.py` `split_markdown_sections` / `canonical_feedback_section` / `merge_rewritten_sections` | The polisher rewrites ONLY the reviewer-addressed sections (one small call each, ≤3 per pass, context of the untouchable rest as Dossier-style "do not repeat") and code merges them deterministically: untouched sections stay byte-identical including header lines → the Nov 8 whole-doc-drift failure mode (polisher dropped the Aegis header while keeping fabricated lines) is structurally impossible. Deterministic link gate refuses link-dropping rewrites; empty/unmatched/cross-cutting targets ("formatting") and refused merges fall back to the improver's full-rewrite chain behind `polisher-full@1` — best-of guard and all routing tests unchanged |
| A36 | Suite 236 → 252 | 15 agent tests + 1 server test | Splitter/canonical/merge (×6), DECLINED variants (×3 incl. improver integration), graph-level version stamps (×1), polisher targeted/full/refused/stamp/gate (×5), server result payload integrity+versions (×1). `_run_with_drafts` gained a `sections=` knob; existing best-of-2 tests pass unchanged (empty sections → full-rewrite path) |

**⚠️ Still hazardous**: the Nov 6 directories + `Optimized_Resume (10).pdf`, and the Nov 8 run directory `539f7c1d…` — do not send either. This round needs BOTH a **FastAPI backend restart** and an **extension reload in `chrome://extensions`** (shared ResultsDashboard source changed). Suggested live validation when the user is ready: rerun the same FortyGuard JD + Aegis context and confirm the Integrity tab shows the declined n8n/MCP requests plus the polisher's targeted section list.


---

## Phase 3 — Feature roadmap 🔲 OPEN (priority order)

### 11 🔶 Before/After diff view — extension side SHIPPED (paste mode), web/server side open
The extension now diff-checks the optimized resume against the original client-side (line LCS + similarity pairing, `frontend/src/components/DiffView.tsx`) whenever the resume was pasted. The remaining gap is an API one: uploaded files are parsed server-side, and `OptimizationResult` never echoes the original — sending it would require the payload change the redesign brief excluded. ATS score delta needs a pre-run score the backend doesn't produce.
- *Server-side completion sketch (needs API change sign-off):* streamer re-emits `original_resume` alongside `optimized_resume`; the same client component consumes it; add `/step2?include_diff=original` so the field is opt-in.
- *Effort:* S (wiring only, once the contract change is approved).

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

### 15 🔶 Production deploy of the new stack
Single-container deploy serving the built frontend + the FastAPI server (uvicorn) — supersedes the Gradio Space instructions in the README. Verify `SKIP_STARTUP_HEALTHCHECK` behavior in startup logs; `.gitattributes` LFS only if PDFs become assets.

---

## Phase 4 — Larger bets 🔶 OPTIONAL

| Bet | Notes |
|-----|-------|
| LangSmith eval set | Replay exported traces + fresh runs into a dataset (score sanity ranges, no fabricated sections, keyword precision). **Must use `client.runs.query()`** — `list_runs()` deprecates after **Jan 31 2027** |
| Dockerfile | Bake LaTeX toolchain for reproducible deploys beside HF Spaces path |
| ✅ Frontend rebuild (FastAPI+React) | **SHIPPED Oct 24–28** — `server/` + `frontend/`, live-probed; see the "Product layer" section above |

---

## Backlog candidates found this session

1. **Semantic-verify retry** (`ats_check_node::_semantic_verify_missing`) — same structured-output transient-failure family as the reviewer; currently fails soft (keeps keywords as missing). Consider the same retry-once pattern once traces show its failure rate.
2. **Top-20 frequency bias** — on the incident JD, `python`/`git`/`llm` lost the top-20 cut to generic words (`workflows`, `ai`); smoke preflight flagged it. Not a defect; a ranking-quality observation for the ATS extractor.
3. **Ubuntu vs Windows coverage drift** — the 84% gate has headroom (measured 87% on Windows); if CI (Linux) trends near the gate, consider measuring coverage locally with the same flags before raising/flooring it.
4. **Light-mode theming decision (frontend)** — the design is dark-only (`frontend/src/index.css` tokens + hardcoded hex in components); decide: proper light token set (recommended) vs defer indefinitely. One design pass, touch every component.
5. **Geist font pairing + motion polish (frontend)** — typography is the system-font stack today; `@fontsource-variable/geist` + `geist-mono` candidates; `motion` is installed but unused — restrained reveal/progress transitions only.
6. **Chrome extension MVP** — decision recorded (README, Oct 28): third client of the same FastAPI API as the web frontend, BYO-key in `chrome.storage.local`, Manifest V3 + side panel reusing the API; content-script JD extraction (LinkedIn/Indeed/Greenhouse/Lever/Workday) is the killer feature. **MVP built Nov 2 — committed as `9e96ba3`; the Nov 3 UI redesign follows in the `feat(extension-ui)` commit.** Next: store assets, scraper fixture tests, permission review, live LinkedIn smoke pass.

## Production-readiness checklist (from full-history trace data, Oct 22)

Evidence base: 871 runs / 73 traces (2026-08-02 → 2026-10-21), 159 LLM runs, 12.4K avg tokens per full trace, 9 errored child runs (2 traces), 0 feedback records.

**Proven by traces (keep as-is):** zero hard crashes in 10 weeks (retry + grace paths absorb every observed failure); no runaway refinement loops (improver count == reviewer count == 43, cycle cap holds); stable token envelope per full run (10.4–14.1K); deterministic LLM-independence of quality guards (ATS engine, hallucination filter, score math); latest-10 reviewer scores 78–85.

**Gaps to close before multi-user production:**

| # | Gap | Trace evidence | Action |
|---|-----|----------------|--------|
| 1 ✅ | Structured-output retry is unproven for its target failure | Trace `01a15f6d` (Oct 21 17:06): same-payload retry folded identically 3s apart | **DONE `27260ee`** — `_salvage_review_output()` parses the model's prose answer (`failed_generation`) instead of retrying identical bytes: markdown/unicode-space normalization, `≈ 80 / 100`-style extraction with combined-weighted line preference, 0–100 clamp, `body → response → repr` fallback chain. Salvage short-circuits the retry; unparseable failures keep the 1-retry + `review_failed` grace path (tested) |
| 2 ✅ | No monitoring signal | 0 LangSmith feedback records in the whole project | **DONE `1025ff2`** — `log_pipeline_feedback()` emits `composite_score` (0–100) + `review_failed` (0/1, ATS-only comment) to the pinned trace root on every completed Step-2 run (LangGraph reads `config['run_id']`). Dashboard alert rule: filter `composite_score < 78` or `review_failed = 1`; salvage now reports a real LLM score instead of the ATS-only floor |
| 3 | Load/concurrency unproven | All 73 traces are single-user sessions; the fallback chain (429s) never engaged | Soak test: 3–5 concurrent sessions ≈ 30 min; confirm per-request UUID output dirs and SQLite checkpointer behave |
| 4 ✅ | PII policy stops at emails/phones in inputs | Names survive redaction and appear in outputs (letters) | **DONE `07f5582`** — candidate names register + mask: `candidate_name_hint()` (label-aware, name-shaped first line), registered by loader & pre-registered by `app.py` before the loader run traces its input; full-name + ≥4-char token masking (word-boundary, case-insensitive) and `_redact_pii` now also `hide_outputs`. `REDACT_NAMES` env/CI-variable masks older data during re-exports. Resume *content* itself is still traced — hence backups guard against public repos (below) |
| 5 | New features have zero traced runs | DOCX path, startup health check, coverage of `failed_generation` salvage: no occurrences in export | One traced smoke run per new path when shipped (playbook item) — next export should show a salvaged-review trace with real feedback records |
| 6 ✅ | Trace history not backed up on a schedule | Exports are manual; free-tier retention applies | **DONE `0767502`** — `.github/workflows/trace_backup.yml`: weekly (Sat 22:00 UTC, `workflow_dispatch` too) export + offline report uploaded as a 90-day artifact. Guard aborts on public repos (traces hold resume PII); requires the `LANGSMITH_API_KEY` Actions secret (optional `REDACT_NAMES` variable); local `langsmith_export/` JSONL remains the durable master. Reminder: GitHub auto-disables scheduled workflows after 60 days of repo inactivity |

**Alerting in the LangSmith dashboard:** filter feedback `key = composite_score` / `key = review_failed` on project *resume-optimizer*. `review_failed = 1` - or the absence of a `composite_score` record (pipeline crashed before completion) - marks a run for review.


## Verification playbook

```bash
# Full suite (no network / no API cost)
venv\Scripts\python.exe -m pytest tests -q

# Coverage check (mirrors CI gate)
venv\Scripts\python.exe -m pytest tests -q --cov=agent --cov=app --cov=server --cov-report=term --cov-fail-under=84

# Frontend + extension checks / build
cd frontend
npm run lint && npx tsc -b && npm run typecheck:extension
npm run build:extension   # regenerates extension/ and extension/dist/ (Load unpacked)

# Live traced smoke run (one real pipeline run; exports trace to traces/)
# script kept in %TEMP%\opencode\smoke_run_traced.py during the session

# Bulk-export ALL LangSmith runs of the project + offline analysis
venv\Scripts\python.exe scripts\export_langsmith_traces.py
venv\Scripts\python.exe scripts\analyze_langsmith_traces.py

# Same backup+analysis runs automatically every Saturday 22:00 UTC via
# .github/workflows/trace_backup.yml (Actions tab -> "Weekly LangSmith trace
# backup" -> Run workflow for a manual run). Requires the repo secret
# LANGSMITH_API_KEY; artifacts expire after 90 days - the local
# langsmith_export/ JSONL stays the durable master.

# SQLite checkpointer persistence (two fresh processes)
# scripts kept in %TEMP%\opencode\ckpt_put.py / ckpt_get.py
```

**Definition of done for any new phase item:** feature code + tests green in the full suite, coverage gate still passing, live smoke run re-verified when the pipeline's LLM path changes, README changelog row (change/why/impact) added, this file's status updated.
