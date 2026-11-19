# Chrome Web Store listing — Job Application Optimizer

## Store name
Job Application Optimizer - AI Resume Agent

## Short description (≤132 characters)
Tailor your resume to any job posting, check ATS fit, and export a one-page PDF and cover letter — from a Chrome side panel.

## Category
Productivity

## Full description
Optimize your resume for the job you actually want, without leaving the job board.

Job Application Optimizer is a Chrome side panel that reads the posting on the page and helps you tailor an existing resume to it. It does not invent experience: missing skills are surfaced as honest gaps, and unevidenced metrics are declined.

What it does
- One-click capture of job descriptions from LinkedIn, Greenhouse, Lever, Indeed, Workday and Ashby (plus a generic fallback for other boards).
- Deterministic ATS keyword and requirement matching, with a weighted score and an honest keyword-gap breakdown.
- AI-assisted resume and cover-letter rewrites grounded in your original resume and any notes you add.
- One-page PDF + DOCX resume, cover-letter PDF, and LaTeX source you can keep editing.
- Before/after diff, bullet-impact and provenance views, application history, and artifact re-download.

Privacy by design
- Resume text, the captured posting and your settings are stored locally in your browser (chrome.storage.local).
- Data is sent only to the backend URL you configure — localhost by default, or a cloud server you choose. No analytics, no ads, no data sale.
- Use Settings → "Forget my data" to erase the saved resume, captured posting and onboarding state.
- Host access for a cloud backend is opt-in: Chrome asks for it when you save the URL. The extension has no blanket https access.

Before installing
You need a running backend (see the project README) or a hosted deployment, and a Groq API key. You can use your own key (BYOK); it is stored locally and sent only to your backend.

## Assets
- `promo-tile.png` — 440 × 280 store tile.
- `screenshot-1.png` — the optimizer input view, shared web/extension React UI with demo data (1280 × 800).
- `screenshot-2.png` — the connection settings modal from the same build (1280 × 800).

## Permission justification (for the store review form)
- `sidePanel`: renders the optimizer UI beside the job posting.
- `activeTab`, `scripting`: inject the floating "Optimize Resume" button on the job page you are viewing.
- `storage`: persists your resume draft, settings and captured posting locally.
- `http://localhost/*`, `http://127.0.0.1/*`: the default local backend (any port).
- Optional `https://*/*`, `http://*/*`: only requested when you configure a cloud backend, so the installed extension has no broad host access by default.
