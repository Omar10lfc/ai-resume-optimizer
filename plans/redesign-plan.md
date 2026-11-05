# Extension UI Redesign — Implementation Plan

Design read: narrow-panel productivity tool for job seekers, calm/trustworthy language, existing cyan-on-near-black identity kept, one accent locked, Tailwind v4 tokens + Phosphor icons.

Dials: VARIANCE 3 (utility panel — low), MOTION 3 (calm; status pulses + entry fades only), DENSITY 6 (data-heavy flows need compact rows).

## Constraints honored
- No changes to api.ts, extension/src/*, manifest, backend, message passing.
- No new deps. React 19 + Tailwind v4 + Phosphor + motion available (motion NOT needed: CSS transitions suffice; less bundle drag).
- No inline scripts anywhere.

## Files to write
1. frontend/src/index.css — full token system (light+dark via :root + [data-theme] + prefers-color-scheme), @theme inline mapping, base layer, component primitives (.btn-*), motion with reduced-motion guards.
2. frontend/index.html — drop hardcoded class="dark"; theme-aware pre-paint <style> (style tag, not script); container wrapper.
3. frontend/src/main.tsx — initTheme() before render.
4. frontend/src/lib/theme.ts — pref system/system|light/dark; chrome.storage.local "uiTheme" / web localStorage "ui_theme"; live follows matchMedia + storage events; hook.
5. frontend/src/lib/gaps.ts — parse missing_skills → chips; classify tool/concept/general; composeStep2Notes(notes, kept, removed): notes = free text extended by `- skill: context` lines for kept chips and one "do not feature removed skills" line. (Server prompt at improver.py:56 consumes notes as guidance; no API change.)
6. frontend/src/lib/errors.ts — classifyError against REAL server strings: "Unauthorized" (401), "Resume file exceeds the N MB limit." (413), "Pipeline queue is full (N waiting)." (SSE error), "Please provide a Job Description", "Unsupported resume file type", "Unknown session", groq timeouts/rate-limits via substrings, fetch failures ("Failed to fetch"/"NetworkError"/"backend offline").
7. components/ToastStack.tsx — useToasts hook (push/dismiss, 5s ttl) + <ToastStack> with aria-live=polite.
8. components/Skeleton.tsx — shimmering block + ResultsSkeleton (gauge+bar+rows reserved height, no CLS).
9. components/ScoreGauge.tsx — SVG ring gauge; composite = score; ATS bar = ats_percentage; quality bar = (score - w_ats*ats)/w_llm clamped 0–100; threshold tick at 80 (server's threshold).
10. components/DiffView.tsx — paste-mode only: line-level LCS diff + similarity pairing for rewritten bullets; degrades to aligned rows without invented matches; hidden unless original text exists.
11. components/Markdown.tsx — shared ReactMarkdown config, token-styled.
12. components/Onboarding.tsx — <=3 step modal (intro / 2-step flow / local data privacy); flag "onboardedAt" in storage; skip + step dots; Escape closes.
13. components/Header.tsx — brand kept; health pill: popover → focusable details trigger (chrome友好 a11y); theme segmented control (Monitor/Sun/Moon); Settings + Reset: equalize labels; Disable reset mid-run.
14. components/InputSection.tsx — detected-job card (editable title/company via jobMeta, fixable description always in textarea); upload zone: real <button> trigger, low-effort validation (accept ext; >5MB inline error MAX=5MB from config.py), clear button; two-point readiness checklist under CTA.
15. components/HumanReviewModal.tsx — GapEditor: 3 groups (tool/concept/general), per-chip remove (X) with single-level UNDO, per-chip note input, rename, keyboard (Delete=remove, U=undo, Enter=rename, N=note), summary counts; SETS userNotes ON SUBMIT via composeStep2Notes (prop unchanged).
16. components/StepTimeline.tsx — keep nodeLog-derived state logic (server truth), + elapsed timer (stops on done/fail), "usually <X min" static hints on nodes (marked usually/estimate), failure banner at last-active node if error arrives, staleness nudge after 90s without events ("long documents take up to ~10 min").
17. components/ResultsDashboard.tsx — ScoreGauge head; downloads as rows with kind/icons; ARIA tablist (arrow-key nav, roving tabindex); tab content lazy-rendered; per-tab copy button; "Compare to original" DiffView in Resume tab only when original exists; ATS breakdown section (deterministic ATS + quality share from weights).
18. components/SettingsModal.tsx — tokenized; label htmlFor fixes; Escape/focus-trap; DANGER zone "Forget my data": chrome.storage.local removal of savedResumeText+pendingScrapedJob+onboardedAt (web: saved_resume_text localStorage), two-step confirm.
19. App.tsx — add jobMeta {title,company} (replaces badge string attempt to preserve; store pendingScrapedJob objects), toasts, ErrorState inline with retry (classification-driven), first-run onboarding mount check, debounced (500ms) storage save for resume text, staleness watch, failed-node detection, ERR aria-live handled by ToastStack; keep all behavioral keys untouched.
20. README/NEXT_STEPS — changelog entries.

## Deliberate skips (will be listed in changelog)
- ATS delta in before/after: server has no pre-run score → would need API change (out of scope).
- Per-node ETA countdowns: no real ETA available; static hints only.
- Node-level "failed": server reports global error event only → mark last-active node failed, rest run-level.
- No virtualization for gaps (max ~30 chips fine).

## Validation
- npm run lint (oxlint)
- npx tsc -b (frontend typecheck)
- npm run build (SSPA sanity)
- npm run build:extension (full MV3 bundle + integrity)
- python pytest -q (no backend change expected to affect tests)
