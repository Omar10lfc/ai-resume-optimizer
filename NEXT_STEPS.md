# Next Steps — Job Application Optimizer

Last updated: **Nov 12 2026** · Snapshot: 323 tests passing (~4:05) · coverage 85% (CI gate 84% over agent+app+server) · langgraph stack 1.2.16 · FastAPI `server/` + Vite React `frontend/` shipped & live-probed · **Chrome extension MVP + UI redesign committed · Nov 8: `539f7c1d` stale-chip laundering closed + PDF entry-boundary fix · Nov 10: Dossier-study round landed (QW1 prompt stamps / QW2 decline protocol / QW3 integrity UI / MR1 section-targeted polish) + durable fix (`c7dd521`: emergency model → `qwen/qwen3.8-27b`, all-tiers catalog validation, fast-fallback BYO key) + MR4 over-the-wire fake-LLM test layer + MR2 requirement-fit table (Fit tab + `_NON_EVIDENCE_TERMS` calibration + company-filter `\b` regression) · Nov 11: PDF layout specs A+B calibrated (per-row Skills items + uniform into-heading gaps + name top-margin + rigid contact glue) + MR3 durable reattachable runs (SQLite runs ledger + `GET /api/v1/runs/{session}` + startup recovery + crash-rewind) + QW4 `.env.example` audit + MR5 provider abstraction (uniform schema-repair ladder + per-task token usage + `LLM_BASE_URL`/`LLM_TIMEOUT_SECONDS` knobs) + BB1 Phase A (persistent profile store + application history) + BB1 Phase B (workspace artifacts + history re-download + server-side diff, roadmap item 11 closed) + MR3 recovery-anchor fix (live run `22e943c6`: a real mid-node crash left an interrupted polisher task whose `versions_seen` outranked the rewound scanner anchor, so the unanchored notes write re-anchored onto it and "recovery" resumed at the fan-out shipping score 0 / "No resume generated." — reset + notes now land as ONE `as_node="scanner"` write and the crash-retry redoes the model segment; gap-channel exact dedupe) + cover-letter integrity (the same incident's letter fabricated "70%"/"45%": truthfulness gate `cover-letter@2` + the resume path's unevidenced-metric scan on the generated letter, dedicated `cover_letter_metrics` channel + own SSE warning + Integrity-tab block) + hanging-indent label rows (live runs `ca102fa5`/`71f879ae`: a Skills row's wrapped tail printed flush-left at the SAME 10.96pt delta as the row-to-row gap, so "Fine-Tuning, Neural Networks" read as the next skill row — label rows now replace the list's one-entry `\parshape` with a two-entry hanging shape indenting continuations 1em, paragraph closed inside the item group where latex.ltx's next-`\item` break can't revert it, `\looseness=0` keeps the shave inert; zero vertical cost, last baseline byte-identical) + date-column stack glue closed (live run `28e2e56f`: `_DATE_RANGE_RE`'s `(.+?)\s*$` tail swallowed post-date pipe fields, gluing tech stacks into the right-aligned date column at x1=735 — 123pt off-page; shared date/tail splitter + pipe-split rejoin + emission guard; 3 stack rows restored, last baseline 718.29 → 760.63, still 1 page) + compact air tiers (tier-1 airy −1.0/−6.0 restores ~4.7pt rule→content air, adopted only when 1 page AND last baseline ≤ the 763.2pt stop; tier-2 tight stays the shipped −1.5/−8.5; probe-compile + winner-copy closes a losing-tier-overwrite hazard) + `" %"` spacing canonized + wire-test concurrent arrival-order flake relaxed to a sorted multiset (flake measured on clean and modified trees alike) + sheet-edge cap (run `124f40cb`: the template's text block bottom sat 6.6pt PAST the 792pt sheet — a page-full base render printed the 4th certificate's glyph tops only while the page count said "1 page", so the compact containment never engaged; both render modes now end their block at 776pt and "1 page" is truthful again — this resume ships tier-1 airy with everything visible) + contact-link preservation (the rewrite dropped the HF profile link — it existed only as a bare label inside a markdown link pointing at LinkedIn; deterministic guard re-injects dropped source-header links on every writer pass + cleans the dangling pipe), suite 317** + round-6 run-pair fixes (runs `bf0f7941`/`03a3ffc1`: balanced middle compact tier -1.5/-7.5 - the with-context resume grew long enough that the airy tier no longer fit one page and the old tight floor shipped ~0.2pt ink under every rule (user: 'no minimal space'); the new tier-2 spends the floor's unclaimed point back and ships +0.14..+1.44pt top-gaps on 1 page, the floor becomes tier-3 last resort; contact-line placeholder sanitation - the draft emitted a literal "N/A" field and the link guard restored right after it; parser orphan salvage - a bullet fragment without its "- " marker was gathered as a header field, took the never-rendered sub slot AND dropped the entry date; mid-sentence lines now render as entry content (joined to the previous bullet or kept as their own item) and trailing date columns survive on >=2-header-line entries; dropped-evidence tripwire - a rewrite that loses a source accomplishment metric (the WER 51.7% claim vanished while its ROUGE-L half survived as the fragment) now surfaces in the guardrail channel, detection-only), suite 323**

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

## Provider hardening — Nov 10 2026 (durable fix for the live 404, own commit `c7dd521`) ✅ LANDED

The live "Optimization failed" run's root cause: a retryable-failure fallthrough (primary → fallback → emergency) landed on `llama-3.1-8b-instant`, which Groq had decommissioned from the catalog. 404 `model_not_found` is not in `_RETRYABLE_EXCEPTIONS`, so it aborted the run — and the startup health check only ever validated the primary tier, so the dead emergency default had sailed through. Durable fix shipped as its own small commit; short-term the same failure is avoided by the `.env` `EMERGENCY_MODEL` override suggested in the incident write-up (not present in `.env`, so the new repo default governs).

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A37 | Emergency model recommissioned + all-tiers catalog validation + fast-fallback BYO key | `agent/config.py` `_EMERGENCY_MODEL` default → `qwen/qwen3.8-27b` (verified on the live catalog via the account key's zero-token `GET /models`); `agent/llms.py` `_groq_models_status` gains `_catalog_gap_warning` — validates ALL three tiers (urllib + httpx branches) and names each missing tier with its `*_MODEL` override; `_create_llm_fast`'s emergency fallback now carries the pinned BYO `api_key` like `_create_llm` already did | Two regressions (`test_groq_models_status_missing_emergency_only_warns` = the incident shape, `test_create_llm_fast_fallback_carries_api_key` exercises the real fallback chain); two health-check fixtures updated to the all-tiers message. Suite 252 → 254. Actionable startup warning now e.g. `missing from this account's catalog: emergency '…' (set EMERGENCY_MODEL)` |

**Restart needed**: FastAPI backend only (agent + tests changed; frontend/extension untouched).


---

## MR4 — Nov 10 2026 (fake-LLM over-the-wire e2e test layer, test-only) ✅ LANDED

The existing server layer fakes (`tests/test_server.py`) monkeypatch the pre-configured chains' `invoke()` — below the chain, the whole `langchain-groq` client stack (httpx → groq SDK → ChatGroq tool calling / structured-output parsing) had zero test coverage. That is exactly the layer that failed in production on Nov 10 (decommissioned model on the wire). MR4 closes the gap with a local fake REST server: the layer was planned in the Dossier study (`DOSSIER_COMPARISON.md` addendum, repo root, intentionally uncommitted) and shipped as tests-only.

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A38 | Fake Groq REST server (`tests/fake_groq_server.py`, stdlib only — no new deps) | Serves the real wire protocol the groq SDK speaks (POST `<base_url>/openai/v1/chat/completions`, GET `/openai/v1/models`) on an ephemeral 127.0.0.1 port (`ThreadingHTTPServer`, keep-alive); branching keyed on the BOUND TOOL first (`ReviewOutput` → score 85 + feedback + `sections=[]`, `SemanticReview` → `verdicts=[]`), then on each node's prompt phrasing (`MISSING SKILLS` / `Rewrite the resume` / `SURGICAL REVISION` / `Cover Letter` / `Preparation Tips`); every accepted completion is recorded `{"path","model","tool","key"}` for arity assertions | Tool-first keying is deliberate: reviewer/semantic prompts embed job text that could match a text branch; "SURGICAL REVISION" is checked before "Rewrite the resume" because the polisher's full-rewrite fallback shares the improver's prompt builder. Unknown bound tools emit an empty-args tool call (loud + visible), unknown paths 404 in the OpenAI error shape |
| A39 | Wire e2e tests (`tests/test_fake_llm_wire.py`, 6 tests, suite 254 → 260) | REAL ChatGroq chains rebuilt with `base_url=` pointing at the fake (the same module-attribute seam `test_server.py`'s fake_llms fixture uses — no pipeline code can tell real from fake) drive BOTH pipeline steps through the real FastAPI SSE app; asserts the fresh step-2 wire sequence `missing_skills → rewrite → tool:ReviewOutput → tool:SemanticReview → rewrite → tool:ReviewOutput → tool:SemanticReview → {cover, prep}` (exact until the concurrent fan-out tail), the checkpointed step-1→step-2 session resume (segment 1 NOT re-run, scanner absent from the wire), the full merged 7-stamp `prompt_versions` payload incl. `polisher-full@1` (sections=[] fallback), artifact emission + downloads, and the fake server's own mechanics (models endpoint, 404s, four-way concurrent completion isolation) | The exact-arity wire contract is the new guard against "the pipeline makes a surprising extra/fewer LLM call" (the Nov 10 accountability gap: the emergency tier only fired after two legitimate calls — wires make such ladders countable). Test-only commit: agent/server unchanged, no restart needed |


---

## MR2 — Nov 10 2026 (deterministic requirement-level fit table) ✅ LANDED

The ATS percentage says how many keywords matched, never which stated requirements are covered — the Dossier study's analytical failure class (front-page score from echoing JD vocabulary with zero requirement coverage) was invisible. MR2 ships the study's `match-requirements.ts` pattern re-implemented on this repo's own matcher machinery: per-requirement classified-in-code rows with human reasons, no new LLM call. Planned in `DOSSIER_COMPARISON.md` (repo root, intentionally uncommitted).

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A40 | Requirement match engine + wiring (`agent/match.py` new module; `agent/state.py` channel; `agent/nodes/ats_check.py` compute + best-of snapshot/ship; `agent/__init__.py` export; `server/runners.py` payload key; `frontend/src/types.ts` + `ResultsDashboard.tsx` conditional Fit tab) | Candidates segmented one-per-line from the boilerplate-stripped JD (bullets/numbered = single cards, headings markers only, plain non-bulleted skill lines don't bleed); vocabulary = the JD's own `_extract_keywords(top_n=20)`; rows `{requirement, status: covered/partial/gap, evidence, missing, reason}` in JD order (cap 12, normalized-key dedupe); matching reuses the ATS exact/synonym/stem machinery + the pass's semantic reclassification set (no extra LLM call); the best-of guard keeps the WINNER's table so a regressed polish pass cannot ship a table describing the regressed draft; payload carries `requirement_matches` and the dashboard renders status badges + counts chips + copyable fit text | Probe-driven calibration (Nov 10): role-filler deny-list `_NON_EVIDENCE_TERMS` (`engineer`/`developer`/`data`/`agent`... — they tell users to "add engineer"); prose sentences need ≥1 Tool/Concept tier term so "TechCorp builds data platforms." never poses as a requirement; bullets qualify on any post-filter evidence; the calibrated ATS percentage is untouched throughout |
| A41 | Extractor regression fix + golden floors (`agent/ats.py` `_extract_company_names`; `tests/fixtures/golden_ats.json`; suite 260 → 273) | The company heuristic `X develops/builds...` lacked a trailing word boundary — "3+ years of Python development experience" prefix-matched `develop|` and captured `Python` as a company name, removing it from the ENTIRE ATS vocabulary (surfaced when the MR2 table silently lost its Python requirement card). Trailing `\b` restores the skill; true brand detection (`FortyGuard builds`, `Support builds support`) verified unchanged by the existing suite | 13 new tests: table unit block (8) + graph best-of wiring (1) in `test_resume_agent.py` (144 → 154), server payload (1) in `test_server.py` (36 → 37), extended wire e2e assertions, golden table floors (2: ≥8 cards / ≥6 covered, additive fixture keys) in `test_golden_ats.py` (6 → 8). Needs FastAPI restart + `chrome://extensions` reload (shared `ResultsDashboard` source) |


---

## Layout specs A+B — Nov 11 2026 (PDF vertical-spacing calibration) ✅ LANDED

The shipped resume PDFs crammed their vertical rhythm: section headings printed as little as 4.6pt under the previous section's last line, Skills rows broke stranded tails that `\looseness` could never shave (one `\\`-joined paragraph), and the name sat 7.3pt from the page's top edge — a print-clip risk against ~34pt of bottom margin. Both render modes (airy base + auto-compact) are now probe-calibrated to a single spec, scripted-measured by a new `pdfplumber` acceptance test. Layout-only round: no resume text, bullet order, links, or fonts changed; ASCII `->` arrows and `\href` full URLs preserved.

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A42 | Vertical-spacing calibration (`agent/latex_render.py`, `agent/templates/resume_template.tex`; `tests/test_latex_render.py` renamed 1 + new acceptance test, suite 273 → 274) | **Spec A (rows):** Skills/rows blocks render one `\item` per row with `itemsep=0pt` so each row gets its own `\looseness=-1` shave pass (the old `\\`-joined single item made every row a forced break inside one paragraph). **Spec B (gaps):** every section heading lands 21.4pt (9.4pt gap) under the previous section's last line in BOTH modes regardless of how that section ends — four section-tail classes (bullet list / 2-row tabular / bare project heading / rows block) each carry a pdfplumber-derived `\vspace` token, so run-B's headings read 21.3–21.5 compact and 21.6–24.1 base (base stays airy by design); entry-title boundaries hold the `539f7c1d` ≥5pt gap floor (2.5× the 2.0pt wrapped-line gap) with the 9pt project-title boundary measured gap-based (14.1pt dB − 9pt = 5.1 ✓); the contact block's `\begin{center}` trivlist glue is rigid (`\topsep`/`\partopsep` zeroed in a group — they are rubber glue the page builder exercised ±2pt differently per body content, probe-proven) and the name's `\vspace*` lands ≥14pt from the page top in every case (worst 14.2 for accent-free names; 16.1 measured on run-B). **Budget:** run-B compact reflows to last baseline 760.6 ≤ the 763.2pt bottom-margin stop (1 page, 0 over/underfull), base mode keeps 2 airy pages. Restart the FastAPI backend to serve the new renderer; frontend/extension untouched | All floors scripted: `test_layout_spec_b_acceptance` compiles a fixture exercising every tail class in both modes and asserts heading gaps ≥9pt + compact uniformity ≤1.5pt, entry ≥5pt, stack→bullet ≥0.5pt, no adjacent-line collision (marker-cluster artifacts excluded), name ≥14pt, dates flush with rules ≤1.5pt, `->`/`\href` survival, and the 763.2pt stop; `pdfplumber==0.11.10` added to `requirements-dev.txt` (test-only). Probe-proven glue mechanics: `topsep` is dead at list start (enumitem `before=\vspace` is the 1:1-alive lever), non-linear at list end; `partopsep` causes trivlist side-effects — avoided everywhere. Known survivors, documented not fixed: a stranded Skills row-1 tail cannot be shaved without rewording (text changes are out of scope), and 10pt-bullet-marker vs 9pt-text baseline clustering is a measuring artifact, not layout |


---

## MR3 — Durable, reattachable runs — Nov 11 2026 ✅ LANDED

A process restart (or a mid-run crash / client disconnect) used to strand a run with no record: the SSE stream died, nothing survived outside the LangGraph checkpoint, and the next step-2 call on the same session silently RESUMED the crashed attempt's half-completed state — that is exactly the mechanism behind live runs `29afd91b` / `62725081` (stale-ATS snapshot: shipped ATS 56.2 while the SHIPPED resume recomputed to 37.5/43.8) and the 7.4s score-59 "band-aid" run. MR3 adds a durable runs ledger and an honest recovery protocol, the Dossier study's `recoverInterruptedJobs()` pattern; "one pipeline at a time" serialization semantics are unchanged (recording, not concurrency).

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A43 | Runs ledger + reattach endpoint + startup recovery + crash-rewind (`server/run_store.py` NEW; `agent/config.py` `RUNS_DB_PATH`; `server/runners.py` recording wrappers + recovery; `server/app.py` endpoint + lifespan; `tests/test_server.py` 37 → 44; `tests/test_fake_llm_wire.py` hermetic ledger fixture; suite 274 → 281) | **Ledger:** one SQLite row per session at `RUNS_DB_PATH` (default `OUTPUT_ROOT/runs.sqlite3` — the artifact cleanup only sweeps directories, so the file survives) with status lifecycle `running → awaiting_notes (step 1 done) → done | failed`, plus `superseded` (fresh-session switch); per-step progress (last graph node started), step-1 chips (the durable pristine `missing_skills` source for recovery — the checkpoint's own copy was already overwritten by the crashed attempt's note injection), the error, and for done runs the full result payload; rows older than the 24h artifact retention are purged by every fresh run's `start()`; ALL ledger writes are non-fatal (print + continue). **Reattach:** `GET /api/v1/runs/{session}` (UUID-validated, 404 unknown) returns status/step/error/chips/result so the SPA can re-render after a reload or disconnect — the SPA currently keeps the session only in React state, so the endpoint is the MR3 contract and the reattach flow is a small documented follow-up. **Startup recovery:** `running` rows found at boot are marked `failed` with an actionable retry message (a restart kills mid-run LLM work; the leftover checkpoint must never be silently resumed), `awaiting_notes` rows survive — the review checkpoint is durable by design. **Mid-pipeline recovery:** step 2 on a thread whose `next` is past the improver emits a `Recovered an interrupted run` warning, re-runs a crashed scanner if needed (one fast-model call), then rewinds via `update_state(_recovery_reset(), as_node="scanner")` — every per-run channel (score/ats/best_result/integrity paths/notes/chips) is reset so the redone segment recomputes scoring on ITS OWN drafts while loader outputs (resume_text/job_text/output_dir) are kept (the uploaded temp file is already deleted; `prompt_versions` is reducer-merged and its stamps are re-emitted identically). **Recording:** both runners are wrapped by `_recorded` — client disconnects and crashes are RECORDED (never masked) via `GeneratorExit`/exception backstops | **Notes injection (probe-calibrated Nov 11):** providing the notes as graph INPUT re-enters the graph from the entry point — the apply-run re-ran loader+scanner (one extra fast-model call per step 2) and the re-run scanner CLOBBERED the user's edited gap lines with fresh chips; caught by the MR4 wire tests' exact call-sequence assertion. The write is therefore exactly one `update_state` (the proven mechanism), and trace visibility of the injected notes — lost because `update_state` is not a runnable invocation (the `29afd91b`/`62725081` forensics gap: "what did the user actually ask?" was unanswerable) — is restored by a `notes_injection` side run whose INPUT is the split notes payload, PII-redacted by the tracer's `hide_inputs` hook like every other run. Restart the FastAPI backend to load the recovery + ledger; frontend/extension untouched |

---

## QW4 — Documented `.env.example` — Nov 11 2026 ✅ LANDED

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A44 | Environment template audit (`.env.example`; docs-only round, no suite change) | Every env var the code reads is now documented with its REAL default (audited by grepping every `os.environ`/`os.getenv` read across `agent/`, `server/`, root `app.py`, and `scripts/` — 20 vars total). Two stale defaults fixed: `EMERGENCY_MODEL=llama-3.1-8b-instant` → `qwen/qwen3.8-27b` (the decommissioned model from the Nov 10 live-404 incident; the fallback ladder + startup catalog validation now documented), and `MAX_ITERATIONS=1` → `2` (the Nov 5 best-of change). Two undocumented vars added: `RUNS_DB_PATH` (MR3 runs ledger, with the resume-content-at-rest note and the strict-privacy pointer) and `SKIP_STARTUP_HEALTHCHECK` (offline dev / CI opt-out). The Dossier study's "generate from the config module" idea is deliberately NOT adopted — a generator script adds a build step for a file that changes once a quarter; the audit-against-grep procedure is documented here instead | No behavior change: `.env.example` is a template; README already links it as the source of truth; CI does not reference it |

---

## MR5 — Provider abstraction — Nov 11 2026 ✅ LANDED

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A45 | Uniform schema-repair ladder + per-task token usage + endpoint knobs (`agent/provider.py` NEW; `agent/config.py`, `agent/llms.py`, `agent/nodes/reviewer.py`, `agent/nodes/ats_check.py`, `server/runners.py` wired; `tests/test_resume_agent.py` 154 → 169, `tests/test_server.py` payload assert, `tests/test_fake_llm_wire.py` exact usage; suite 281 → 296) | **Ladder** (`invoke_structured_with_repair`): direct invoke → ONE second attempt — a schema REPAIR showing the model its own `failed_generation` prose (as an AIMessage, never pasted into instructions) plus the required schema when prose exists, else the plain same-payload retry for the transient class — → caller-supplied prose salvage → raise. Max TWO LLM calls (pre-MR5 cap preserved). Applied to BOTH structured-output call sites that hit the "Tool choice is required" flake class: reviewer (repair recovers the full structured result incl. `sections`, which prose salvage loses; salvage stays the free deterministic fallback) and the ATS semantic check (degrade direction unchanged: keywords stay missing). `extract_failed_generation` moved from `nodes/reviewer.py` to `provider.py` (re-exported for compatibility). **Usage accounting** (`TokenUsageTracker`): one tracker per runner attempt, attached via the new `callbacks=` param on `_thread_config` (fresh list — the shared `TRACE_CALLBACKS` is never mutated), thread-safe for the cover/interview fan-out; reads the OpenAI-shaped `llm_output["token_usage"]` else langchain-core's `usage_metadata`; rides the step-1 `done` event and the step-2 `result` payload (lands in the runs ledger for free). Wire-proven exact: step 1 `done` 1/1/2 (llm_calls 1), fresh step 2 `result` 9/9/18 (llm_calls 9). **Knobs**: `LLM_BASE_URL` (any OpenAI-compatible endpoint rides the MR4-proven ChatGroq `base_url` seam; key-free local Ollama/LM Studio relaxes the import-time `GROQ_API_KEY` requirement; `reasoning_effort` omitted for custom endpoints; `GROQ_API_KEY` doubles as the generic key slot) and `LLM_TIMEOUT_SECONDS` (default 120 → `request_timeout=`, replacing the SDK's 10-minute hang window), both via the single `_chat_kwargs()` construction point; health check derives `<base>/models`, skips the Groq-catalog tier-gap warning for custom endpoints, and keeps the official-API messages byte-identical | **Not disturbed:** the `RunnableWithFallbacks` tracing shape (each ladder attempt is a normal structured-output invoke on the same chain — LangSmith trees stay comparable), the BYO-key seam (`X-Groq-Api-Key` → `_KeyScopedLLM`), and the existing test call-count contracts (`test_reviewer_retry_recovers_from_transient_failure` attempt 2 = plain retry; `test_reviewer_retry_gives_up_after_two_attempts` calls==2; `test_salvage_skipped_when_generation_unparseable` calls==2). **Changed:** `test_salvage_parses_score_from_failed_generation` calls 1 → 2 — repair precedes salvage now (recovers `sections` that salvage loses; the docstring documents the superseded identical-bytes rationale). Empirical probe first: the tracker's wire behavior verified against the fake server before the exact totals were asserted |

---

## BB1 Phase A — Persistent profile store + application history — Nov 11 2026 ✅ LANDED

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A46 | Application history (`server/profile_store.py` NEW; `agent/config.py` `PROFILE_DB_PATH`/`PROFILE_DB_ENABLED`; `server/runners.py` auto-record; `server/app.py` list/detail/delete endpoints; `frontend/src/components/HistoryPanel.tsx` NEW + `App.tsx`/`types.ts`/`api.ts`; `tests/test_server.py` 44 → 48; wire e2e recording assert; suite 296 → 300) | **Store:** SQLite `applications` + `runs` tables, the Dossier study's Application record on this repo's own `run_store` pattern. Applications dedupe on the normalized (JD, resume) SHA pair — re-optimizing the same posting attaches a new attempt instead of forking duplicates (history groups itself); title = the JD's first non-empty line (120 chars); each run keeps score / ATS / LLM score / review-failed + the FULL result payload (`requirement_matches`, `prompt_versions`, `token_usage`, artifacts) in `result_json` exactly like the ledger's reattach payload; FK cascade delete; non-fatal writes (warn + continue) like the ledger. **Posture change, deliberate:** PERSISTENT — no retention sweep; the evidence texts at rest ARE the point (the `%TEMP%` artifacts evaporate in 24h, history does not). Default `data/profile.sqlite3` under the project root (gitignored, 0700 dirs); `PROFILE_DB_PATH=off` restores the old ephemeral posture (no file created, no recording). **Recording:** auto on every completed step-2 run (zero UI, the study's GET-only-import slice) with the PARSED evidence texts from the AUTHORITATIVE SNAPSHOT — `final_state` only accumulates one segment's deltas, so on the step1→step2 resume path the loader outputs (which ran during step 1) must come from `agent_app.get_state()`, not the in-loop accumulator (this was the wiring bug caught before any run). **API:** `GET /api/v1/applications` (newest-first summaries, no evidence text, limit 1–200), `GET /api/v1/applications/{id}` (evidence texts + runs newest-first, payloads parsed), `DELETE /api/v1/applications/{id}`. **SPA:** HistoryPanel — collapsed by default (fetch on first expand, landing page stays quiet), per-application title/attempts/best-score/last-run, expands to per-run score/ATS/LLM badges + `review salvaged` marker + covered/partial/gap chips; delete + refresh; auth/CORS honored via the existing settings seam | **Deferred to Phase B (documented, not dropped):** artifact relocation from `%TEMP%` under the workspace, document versions/undo-restore, profile editor, server-side diff-against-original payload change (item 11). Both test harnesses (test_server + wire) made hermetic for the profile DB in their autouse fixtures |

---

## BB1 Phase B — Workspace artifacts + history re-download + server-side diff — Nov 11 2026 ✅ LANDED

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| A47 | Workspace artifacts + regenerate endpoint + diff echo (`agent/config.py` `OUTPUT_ROOT`; `agent/nodes/pdf_exporter.py` factored `export_documents()`; `server/app.py` regenerate endpoint + `include_diff` param; `server/runners.py` `include_original`; `server/profile_store.py` at-rest filter; `frontend/src/api.ts`/`App.tsx`/`types.ts`/`ResultsDashboard.tsx`/`HistoryPanel.tsx`; `.env.example` `OUTPUT_ROOT` doc; `tests/test_server.py` 48 → 53; suite 300 → 305) | **PB1 — artifacts under the workspace:** `OUTPUT_ROOT` defaults to `<project>/data/artifacts` (inside the gitignored `data/` dir, 0700 dirs) instead of `%TEMP%\resume_optimizer_output` — OS temp cleaners could vanish files mid-retention-window, and `%TEMP%` was invisible to the user. `OUTPUT_ROOT` env knob overrides (documented in `.env.example`, previously an undocumented ghost knob); the runs ledger follows automatically (defaults to `OUTPUT_ROOT/runs.sqlite3`) and the 24h sweep semantics are unchanged (directories only — the ledger file survives). **PB2 — history re-download:** the exporter's core is factored into `export_documents(resume_text, cover_text, request_dir)` (node + endpoint share ONE implementation); `GET /api/v1/applications/{app_id}/runs/{run_id}/artifacts/{kind}` for `resume-pdf`/`resume-docx`/`resume-tex`/`cover-pdf` re-renders from the STORED text of a past run (the evidence outlives the 24h artifact sweep — the honest versions/restore), writing both documents into a fresh UUID request dir left for the regular sweep. Placeholder strings ("No resume generated.") are rejected as non-content; 404 on unknown kind/app/run and cross-app run ids; **honest semantics: regenerated through the CURRENT template — may differ slightly from the original run's files** (surfaced on the UI chips' tooltip). HistoryPanel gets per-run Resume PDF / Cover PDF download chips (DOCX/TEX endpoint-only — no UI button). **PB3 — server-side diff (roadmap item 11 closed):** `POST /api/v1/step2?include_diff=original` echoes `original_resume` — the PARSED original from the authoritative snapshot (on the step1→step2 resume path `final_state` never holds it); `record_run` filters the key back OUT of `result_json` (resume_text is already the application's evidence column — no at-rest duplication; the 24h runs ledger keeps the wholesale payload as always). The SPA always requests the echo and falls back to the client-held text in paste mode (`result.original_resume ?? pasteText`); the extension's paste-only contract is untouched | Tests: `OUTPUT_ROOT` default shape (env-knob escape), regenerate roundtrip from a real pipeline run (content types + `%PDF` magic bytes via LaTeX and the HTML fallback path, `.tex` written before compilation regardless of pdflatex, DOCX 200-or-404 with pandoc availability, ≥2 fresh request dirs), regenerate 404 validation (unknown kind/app/run, placeholder texts, cross-app run id), `include_diff=original` exact echo + at-rest non-duplication + opt-out absence. One existing spy updated for the new runner kwarg (`**kwargs` forwarding). Restart the FastAPI backend + reload the frontend; the first run after upgrade writes under `data/artifacts/` — the old `%TEMP%` location is simply abandoned (nothing migrates; history re-download regenerates on demand) |

---

## Phase 3 — Feature roadmap 🔲 OPEN (priority order)

### 11 ✅ Before/After diff view — SHIPPED (extension paste mode Nov 7 + server-side echo Nov 11)
Fully closed by BB1 Phase B: `POST /api/v1/step2?include_diff=original` echoes the parsed `original_resume` in the result payload, so `DiffView` works for uploads too (the same client component consumes it, exactly as this sketch proposed). Paste mode still uses the client-held text as the primary source. ATS score delta remains skipped by design — the echo feeds the diff view, not the scorer (a pre-run upload score would require a separate scoring pass).

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
