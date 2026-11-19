# LinkedIn smoke pass — manual checklist

Automated coverage that already exists (run `cd frontend && npm run test:extension`):
- LinkedIn rule matches only `/jobs/` URLs (feed URLs fall through to the generic scraper).
- Title/company/location/description selectors, selector priority, and upsell-footer stripping.
- Missing-description fall-through, generic fallback, and a non-matching URL case.

A **live** pass cannot be completed from a headless/CI environment: unauthenticated requests to `linkedin.com/jobs/` are rejected by LinkedIn (HTTP 429 from this environment; a bogus `/jobs/view/123` returns 404). It needs a logged-in Chrome profile with the unpacked extension loaded.

## Steps

1. `cd frontend && npm run build:extension`, then `chrome://extensions` → Developer mode → **Load unpacked** → select `extension/`.
2. In the same Chrome profile, log in to LinkedIn and open a **single job posting** (URL must contain `/jobs/view/`). Feed and search-results pages are not supported by the LinkedIn rule.
3. Wait for the floating **“Optimize Resume”** button (injected at 1.5s, retried at 3.5s, and on SPA navigation).
4. Click it. Expected:
   - the side panel opens;
   - the job form shows the *Imported Job Posting* badge with the real title/company;
   - the description is populated and contains no “Job search faster with Premium” upsell text;
   - `chrome.storage.local.pendingScrapedJob` is consumed/cleared by the panel.
5. Click the toolbar icon on another job page: the panel opens even when the floating button is absent.
6. Run **Step 1** and **Step 2** against a live backend; confirm the SSE timeline, ATS/fit tabs and the four artifact downloads.
7. Permission review in `chrome://extensions` → extension **Details** → *Site access*: the installed extension must **not** have access to all sites. Only localhost/127.0.0.1 are granted at install.
8. In the panel Settings, set a cloud backend URL. Chrome should prompt for that origin only; accept and re-run the health check. Deny it once and confirm the panel reports the backend as offline rather than silently working.

## Known limits
- LinkedIn frequently changes its class names; if the description stays empty, the generic fallback runs and the **Source** badge should read *Generic Web Scraper*.
- LinkedIn job search/feed URLs often require login and may render no posting even when the rule matches; use a direct job view URL.
- The extension never sends the scraped page anywhere except the backend URL you configured (see `store/privacy_policy.md`).
