# Privacy Policy — Job Application Optimizer (Chrome Extension)

Last updated: 19 November 2026

## What the extension stores locally
Using `chrome.storage.local` in your browser profile:

- **Your resume text** (saved draft) and any notes you add for the optimizer.
- **The last captured job posting** from a job board (title, company, description, URL).
- **Connection settings**: backend URL, optional basic-auth username/password, optional Groq API key (BYOK), and UI theme/onboarding flags.

Nothing is stored on our servers by the extension itself. Use **Settings → Forget my data** in the side panel to erase the saved resume, captured posting and onboarding state. Connection settings are deliberately kept — they are not personal data and are needed to reconnect; clear them manually in Settings if you want them gone.

## What is transmitted, and to whom
- Resume text, the job posting and your notes are sent **only to the backend URL you configure in Settings** (default `http://localhost:8000`) when you start an optimization.
- Your optional Groq key and basic-auth credentials are sent in request headers to that same backend only.
- The backend calls the configured LLM provider (Groq by default) to generate the tailored resume and cover letter. Provider-side retention follows that provider's own terms.
- **No analytics, advertising, telemetry, or third-party trackers** are included. We do not sell or share your data.

## Permissions
- `sidePanel` — the optimizer UI.
- `activeTab`, `scripting` — the floating "Optimize Resume" button on the job page you are viewing.
- `storage` — the local persistence listed above.
- `http://localhost/*`, `http://127.0.0.1/*` — the default local backend, any port.
- Optional `https://*/*` and `http://*/*` — access to a cloud backend is requested at runtime when you save its URL. The extension installs **without** blanket host access.

## Job board scraping
The content script reads the visible job description from the page you are on when you click the button. It does not scrape in the background, does not collect browsing history, and does not transmit the page content anywhere except your configured backend.

## Contact
For questions about this extension, use the project's repository issue tracker.
