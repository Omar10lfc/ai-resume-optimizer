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
