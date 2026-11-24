# Docker Smoke Test (docs) — run when a Docker engine is available

Status: **NOT run** — no Docker engine is available in the environment this
was written in (both the `dockerDesktopLinuxEngine` and the `default`
engine pipes are missing; `docker info` fails). This file is the manual
verification plan for the Dockerfile + docker-compose.yml (Item 9).

Prereqs: a running Docker Desktop/Linux daemon, the `desktop-linux` (or
equivalent) context selected (`docker context ls`).

## 1. Build and start

```powershell
# from the repo root
docker build -t job-application-optimizer:dev .
docker compose up -d
```

Expected:

- `docker build` succeeds (see the Dockerfile header for the hardening
  notes: non-root user, TeX Live, Chromium, fonts, `/dev/shm`, pinned
  versions).
- `docker compose up -d` starts the `app` service; `docker ps` shows it
  running. Logs contain the startup line (`[Health] Groq: OK - ...` or a
  "Problem" line — the app still launches) and, if `APP_AUTH_*` is unset,
  the `[Auth] WARNING: APP_AUTH_USERNAME/APP_AUTH_PASSWORD are not set and
  the server is bound to a non-local address (0.0.0.0); ...` message
  (Item 9 warning). With auth set, the warning is absent.

## 2. Verify image basics (non-root user, sandbox on, openin_any=p)

```powershell
docker run --rm job-application-optimizer:dev sh -c `
    "id -u && ls -l /usr/bin/chromium* 2>/dev/null | head -3 && kpsewhich -var-value=openin_any && kpsewhich -var-value=openout_any"
```

Expected:

- The numeric uid is `1000` (the in-image `app` user, `useradd --uid 1000`),
  not 0.
- A chromium executable path is printed.
- `kpsewhich -var-value=openin_any` prints `p` and
  `kpsewhich -var-value=openout_any` prints `p` (set via `ENV` in the
  Dockerfile).

Alternative (app-level, engine-independent, from the suite):
`pytest tests/test_latex_render.py -k "outside_compile_dir or refuses_input_outside"`
(the outside-file refusal tests).

## 3. Verify the container runs as non-root

```powershell
docker compose exec app id -u
```

Expected: `1000` (or the `HOST_UID` you configured in the compose
`user:` line).

## 4. Verify ./data is writable (bind mount under non-root)

```powershell
docker compose exec app touch data/writable-test.txt
docker compose exec app ls -l data/writable-test.txt
docker compose exec app rm -f data/writable-test.txt
```

Expected: the touch/ls/rm succeed — the app user owns the bind mount
(compose `user: "${HOST_UID:-1000}:${HOST_GID:-1000}"`, matching the
Dockerfile's `useradd --uid 1000`; override with `HOST_UID`/`HOST_GID` if
your host uid differs).

## 5. Full job smoke (Step 1 + Step 2) against the running container

Set a fake/real `GROQ_API_KEY` in `.env`, then:

```powershell
# Step 1 (paste the JD text; resumes via the form as well)
curl.exe -s -X POST http://localhost:8000/api/v1/step1 `
    -F "job_description=<JD TEXT>" -F "resume_text=<RESUME TEXT>"
```

Expected: an SSE stream (`event: status` … `event: result`); the result
event carries `missing_skills` (step-1 chips). Then:

```powershell
# Step 2 (same form + notes)
curl.exe -s -X POST http://localhost:8000/api/v1/step2 `
    -F "job_description=<JD TEXT>" -F "resume_text=<RESUME TEXT>" -F "notes=k8s"
```

Expected: SSE stream ending in `event: result`; artifacts appear under
`./data/artifacts/<session>/` (e.g. `Optimized_Resume.pdf`,
`Optimized_Resume.docx` when pandoc is present, `Optimized_Resume.tex`,
`Cover_Letter.pdf`). The files exist and are non-empty. With the default
renderer (`auto`), non-classic themes go through Chromium; the classic
look prints through LaTeX (both engine paths should work — check the
files exist).

## 6. Verify openin_any=p is in effect (A3-style)

```powershell
docker run --rm job-application-optimizer:dev python -c `
    "import os; print(os.environ.get('openin_any'), os.environ.get('openout_any'))"
```

Expected: `p p`.

## 7. Startup warning behavior (Item 9)

- Unset `APP_AUTH_USERNAME`/`APP_AUTH_PASSWORD` in `.env` →
  `docker compose up` logs contain the `[Auth] WARNING: ...` line
  (server/app.py lifespan).
- Set them → restart → the warning is absent and `GET /api/v1/health`
  requires credentials (401 without).

## 8. Health check

```powershell
curl.exe -f http://localhost:8000/api/v1/health
```

Expected: HTTP 200 (or 401 when auth is set). The compose healthcheck uses
the same endpoint; when auth is set, extend it with `-u user:pass` or add a
token.

## 9. Teardown

```powershell
docker compose down -v   # or keep ./data mounted as the data volume
```

## Status line

Written during the follow-up round (Item 4). Status: **NOT run** — see the
top of this file.
