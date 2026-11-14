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

# BB2 (Nov 14 2026) made the HTML/CSS renderer (headless Chromium) the
# auto-selected resume engine when a browser is found. Tests pin the LaTeX
# path instead: every LaTeX-geometry pin (test_latex_render.py) and the
# existing export expectations were written for it, and the suite must not
# depend on a browser binary being installed. The HTML renderer is
# exercised directly in test_html_render.py (engine-selection tests opt in
# explicitly; the scripted acceptance skips when no browser is present).
os.environ["RESUME_RENDERER"] = "latex"
