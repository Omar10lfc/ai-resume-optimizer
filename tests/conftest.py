"""Session-wide environment pins. pytest imports conftest BEFORE collecting
and importing test modules, so this runs before ``agent.config`` reads its
environment."""

import os

# Round 8 made the checkpoint store durable BY DEFAULT (the workspace's
# gitignored data/checkpoints/checkpoints.sqlite3). Tests pin the escape
# hatch instead, restoring the historical in-memory checkpointer: sessions
# stay hermetic (no workspace writes at graphs-import time, no cross-test
# state leakage through a shared db file). The SQLite path is exercised
# directly against tmp_path in test_resume_agent.py.
os.environ["CHECKPOINTER_DB_PATH"] = "off"

# BB2 (Nov 14 2026) routed the default to the HTML/CSS renderer (headless
# Chromium) when a browser is found; Nov 15 2026 made auto look-driven
# (the classic look prints LaTeX again). Tests pin the LaTeX path either
# way: every LaTeX-geometry pin (test_latex_render.py) and the
# existing export expectations were written for it, and the suite must not
# depend on a browser binary being installed. The HTML renderer is
# exercised directly in test_html_render.py (engine-selection tests opt in
# explicitly; the scripted acceptance skips when no browser is present).
os.environ["RESUME_RENDERER"] = "latex"

# The repo .env (agent.config load_dotenv()s it at import) carries
# LANGCHAIN_TRACING_V2=true; the tracer activates at agent.tracing import
# time and feedback writes would leave the process. Most modules pin it
# off at the top (historical belt); this pin makes EVERY module order
# hermetic (test_html_render.py never carried the belt - running it first
# activated a real client for the whole run, Nov 15 2026).
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Round 15 (roadmap item 13) made the chat-model family selectable
# (LLM_PROVIDER: groq | openai | anthropic | openai-compatible). Tests pin
# the historical groq engine: every chain/health expectation was written for
# it, and the suite must not depend on which provider a developer's .env
# selects. The openai/anthropic construction paths are exercised directly in
# test_resume_agent.py (the provider matrix opts in explicitly).
os.environ["LLM_PROVIDER"] = "groq"

# Unit and integration tests use fake/monkeypatched LLMs or mocked HTTP servers;
# seeding a dummy key prevents import-time config failures on CI runners or
# clean checkouts without a local .env file.
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key_for_tests")
