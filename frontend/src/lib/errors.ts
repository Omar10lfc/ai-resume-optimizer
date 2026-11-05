/**
 * Error taxonomy.
 *
 * The server surfaces failures as HTTP detail JSON (app.py), SSE `error`
 * events (queue-full, node exceptions), and fetch rejections (offline).
 * api.ts merely forwards the raw string, so classification happens here,
 * client-side, by matching the exact strings the backend produces.
 *
 * Every class pairs a short title with a plain-language detail and, where
 * meaningful, the label of the action the user should take next.

 * Matching is intentionally substring-based (case-insensitive) — the SSE
 * error channel wraps node exceptions verbatim, so exact-match parsing
 * would be brittle and provide no value.
 */

export type ErrorKind =
  | "auth" // 401 "Unauthorized" — Basic Auth credentials missing/expired
  | "offline" // fetch failed / connection refused — server not running or URL wrong
  | "busy" // "Pipeline queue is full (N waiting). Try again shortly."
  | "rate_limit" // Groq 429 / free-tier quota / provider rate limit
  | "too_large" // "Resume file exceeds the N MB limit." (413)
  | "unsupported_file" // "Unsupported resume file type (PDF or DOCX only)."
  | "no_jd" // "Please provide a Job Description (text or URL)." / capture found nothing
  | "no_resume" // "Please upload a resume PDF or paste your resume text."
  | "session" // "Unknown session." / "Invalid session id." — server restarted or SID stale
  | "timeout" // Groq APITimeoutError / compilation timed out
  | "unknown";

export interface ErrorInfo {
  kind: ErrorKind;
  title: string;
  detail: string;
  action?: string;
}

const PATTERNS: Array<{
  kind: ErrorKind;
  test: RegExp;
  title: string;
  detail: string;
  action?: string;
}> = [
  {
    kind: "auth",
    test: /401|unauthori[sz]ed|basic auth|invalid credentials/i,
    title: "Sign-in required",
    detail: "The server rejected the request because no valid credentials were attached. If this server uses Basic Auth, add the username and password in Settings.",
    action: "Open Settings",
  },
  {
    kind: "busy",
    test: /queue is full|too many concurrent|server is busy/i,
    title: "Server is busy",
    detail: "Another run is already in progress and the queue is full. Usually clears within a minute.",
    action: "Try again in a minute",
  },
  {
    kind: "rate_limit",
    test: /429|rate.?limit|quota|too many requests/i,
    title: "Rate limited",
    detail: "The AI provider refused the request for hitting its request limit. On free tiers the quota resets within a minute.",
    action: "Wait a moment and retry",
  },
  {
    kind: "too_large",
    test: /exceeds the .*mb|file is too large|too large/i,
    title: "File too large",
    detail: "The resume file is larger than the server limit. Export a lighter PDF or paste the resume text instead.",
  },
  {
    kind: "unsupported_file",
    test: /unsupported resume|pdf or docx only/i,
    title: "Unsupported file",
    detail: "The server accepts PDF and DOCX files only. Upload one of those or paste the resume text.",
  },
  {
    kind: "no_jd",
    test: /provide a job description|job description is empty|no job posting|couldn(?:'t| not) (?:find|extract).*job/i,
    title: "No job description",
    detail: "The job posting text arrived empty or too short. Paste the job description text or open the posting as a URL.",
  },
  {
    kind: "no_resume",
    test: /upload a resume pdf|paste your resume|resume is empty/i,
    title: "No resume",
    detail: "The server received no resume. Upload a PDF/DOCX file or paste your resume text.",
  },
  {
    kind: "session",
    test: /unknown session|invalid session/i,
    title: "Session expired",
    detail: "The server no longer knows this run — it was likely restarted between steps. Start over with a fresh Step 1 scan.",
    action: "Start over",
  },
  {
    kind: "timeout",
    test: /timed? ?out|timeout/i,
    title: "The run timed out",
    detail: "The provider didn't respond in time. This usually succeeds on retry, especially with a shorter resume.",
    action: "Retry",
  },
  {
    kind: "offline",
    test: /failed to fetch|networkerror|load failed|err_connection|econnrefused|getaddrinfo|backend offline|socket hang up/i,
    title: "Can't reach the backend",
    detail: "The extension couldn't connect to the server. Make sure it's running, and confirm the URL in Settings matches (default: http://localhost:8000).",
    action: "Open Settings",
  },
];

export function classifyError(raw: string): ErrorInfo {
  const matched = PATTERNS.find((p) => p.test.test(raw));
  if (matched) {
    return { kind: matched.kind, title: matched.title, detail: matched.detail, action: matched.action };
  }
  const trimmed = (raw || "").trim();
  return {
    kind: "unknown",
    title: "Something went wrong",
    detail: trimmed.length > 0 && trimmed.length < 300 ? trimmed : "The run failed unexpectedly. Check the server logs or retry.",
    action: "Retry",
  };
}
