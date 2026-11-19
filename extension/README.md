# Browser Extension (Chrome, Manifest V3)

Wraps the existing Vite + React frontend as the extension's **side panel**, adds a
background service worker and a content-script scraper that captures job
descriptions from job boards (LinkedIn, Greenhouse, Lever, Indeed, Workday, Ashby
+ a generic fallback) with one click.

```
extension/
├── manifest.json          # MV3 spec (sidePanel, storage, activeTab, scripting)
├── src/
│   ├── background.ts      # Service worker: side panel lifecycle + message relay
│   ├── content.ts         # Floating "Optimize Resume" button + page messaging
│   └── sites.ts           # Per-site CSS selector registry + generic fallback
├── index.html / sidepanel.html / background.js / content.js / assets/
│                          # Build artifacts mirrored from extension/dist (gitignored)
├── icons/                 # Toolbar + store icons
└── store/                 # Chrome Web Store assets, listing, privacy policy, smoke checklist
```

## Build & Load

```bash
cd frontend
npm run build:extension      # builds UI + worker + content script, mirrors to extension/ and extension/dist/
npm run typecheck:extension  # type-only check of extension/src (Vite transpiles without checking)
```

1. `chrome://extensions` → enable Developer mode
2. **Load unpacked** → select `extension/` (or `extension/dist/`)
3. Open any job posting and click the floating **"✦ Optimize Resume"** button,
   or just click the toolbar icon to open the side panel.

## Backend connection (Settings ⚙ in the panel header)

- **Backend URL** defaults to `http://localhost:8000`; point it at a cloud
  deployment for hosted use. Server-side CORS accepts extension origins via
  `EXTENSION_ORIGINS` (see `.env.example`).
- **Custom Groq API key (BYOK)** is sent as the `X-Groq-Api-Key` header. The
  server honors it per request: the pipeline runs against *your* Groq quota,
  and the health check validates *your* key when present.
- **Basic auth** is forwarded when the server has `APP_AUTH_USERNAME` /
  `APP_AUTH_PASSWORD` set.

## Permission model (least privilege)

The extension installs with **no blanket web access**. `host_permissions` only
covers `http://localhost/*` and `http://127.0.0.1/*` (any port) so the default
local backend works immediately. Cloud backends live in
`optional_host_permissions`: saving a non-local backend URL in Settings calls
`chrome.permissions.request` for that origin only, from the user gesture.
If permission is denied, the panel reports the backend as offline instead of
silently working. After upgrading from the old blanket-permission build, open
Settings and re-save your cloud URL once to grant the new optional origin.

The job-board content scripts are declared by explicit `matches` in the
manifest (LinkedIn, Greenhouse, Lever, Indeed, Workday, Ashby), not by a broad
host grant.

## Tests and store assets

```bash
cd frontend
npm run test:extension   # Node fixture tests for sites.ts (no external deps)
```

The Node test runner loads `extension/src/sites.ts` with type stripping and a
small selector-keyed DOM stub: per-board URL matching, selector priority,
field extraction, upsell-footer stripping, and the generic fallback.
`tests/test_extension_readiness.py` guards the manifest permission shape.

`extension/store/` contains the Chrome Web Store submission assets:
`listing.md` (copy + permission justifications), `privacy_policy.md`,
`promo-tile.png` (440×280) and two 1280×800 screenshots. The screenshots were
captured from the built UI with demo data (no real PII). `LINKEDIN_SMOKE.md` is
the manual checklist for a logged-in Chrome pass — a live LinkedIn fetch cannot
be automated from a headless environment (LinkedIn returns 429 to anonymous
requests).

## Verification checklist

- Toolbar click opens the side panel; the floating button opens it and pre-fills
  the job description (badge: *Imported Job Posting*).
- Resume text is persisted (`chrome.storage.local` → `savedResumeText`) and
  restored on next session; cleared text is purged.
- Step 1 / Step 2 stream over SSE from a live backend; PDF/DOCX artifacts
  resolve against the configured backend URL.
- Both scroll/button flows degrade to a red banner when the backend is offline
  (Test Connection in Settings).

> Note: `manifest.json` requires Chrome ≥ 114 (Side Panel API). Re-run
> `npm run build:extension` after editing `frontend/src/**` — the panel app is
> built from the frontend sources.
