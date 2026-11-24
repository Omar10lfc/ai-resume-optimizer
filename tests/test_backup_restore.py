"""Item 11: backup & restore script (scripts/backup_restore.py) + versioned
migration table (agent/helpers.py: ensure_db_migrations, db_schema_version)."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _tmp_stores(monkeypatch, tmp_path):
    """Points the three stores at tmp files (in-process + env for the
    script subprocess) and returns their paths."""
    import agent.config as agent_config
    import agent.graphs as graphs
    profile = tmp_path / "profile.sqlite3"
    runs = tmp_path / "runs.sqlite3"
    ckpt = tmp_path / "ckpt.sqlite3"
    monkeypatch.setattr(agent_config, "PROFILE_DB_PATH", str(profile))
    monkeypatch.setattr(agent_config, "RUNS_DB_PATH", str(runs))
    monkeypatch.setattr(agent_config, "CHECKPOINTER_DB_PATH", str(ckpt))
    monkeypatch.setattr(graphs, "CHECKPOINTER_DB_PATH", str(ckpt))
    return profile, runs, ckpt


def _script_env(profile, runs, ckpt):
    return {**os.environ,
            "PROFILE_DB_PATH": str(profile),
            "RUNS_DB_PATH": str(runs),
            "CHECKPOINTER_DB_PATH": str(ckpt)}


def _run_script(*args, env):
    return subprocess.run([sys.executable, "scripts/backup_restore.py", *args],
                          cwd=REPO, capture_output=True, text=True, env=env)


def test_versioned_migration_table_in_stores(tmp_path, monkeypatch):
    """Item 11: every store carries the versioned _migrations table (v1),
    created by the stores' _connect."""
    from server import profile_store, run_store
    import agent.graphs as graphs
    from agent.helpers import db_schema_version
    profile, runs, ckpt = _tmp_stores(monkeypatch, tmp_path)
    profile_store._connect().close()
    run_store._connect().close()
    checkpointer = graphs._create_checkpointer()
    assert getattr(checkpointer, "conn", None) is not None
    checkpointer.conn.close()
    for p in (profile, runs, ckpt):
        assert db_schema_version(p) == 1, p
        conn = sqlite3.connect(str(p))
        try:
            row = conn.execute(
                "SELECT version, applied_at FROM _migrations").fetchone()
        finally:
            conn.close()
        assert row[0] == 1 and row[1]


def test_backup_restore_roundtrip(tmp_path, monkeypatch):
    """Item 11: backup the three stores via the script, wipe the live
    files, restore --force, assert rows + schema versions come back."""
    from server import profile_store, run_store
    from agent.helpers import db_schema_version
    profile, runs, ckpt = _tmp_stores(monkeypatch, tmp_path)
    # Seed: an application + a ledger row.
    run_id = profile_store.record_run("JD", "Resume", "sess-bk1",
                                      {"type": "done",
                                       "optimized_resume": "X"})
    assert run_id is not None
    app_id = profile_store.list_applications()[0]["id"]
    run_store.start("sess-bk1", "step2")
    assert run_store.get("sess-bk1") is not None
    # Backup via the script.
    backup_dir = tmp_path / "backup"
    env = _script_env(profile, runs, ckpt)
    r = _run_script("backup", "--out", str(backup_dir), env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    manifest = json.loads(
        (backup_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["stores"]) >= {"profile", "runs"}
    assert (backup_dir / "profile.sqlite3").exists()
    assert (backup_dir / "runs.sqlite3").exists()
    assert manifest["stores"]["profile"]["schema_version"] == 1
    # Wipe the live files (the reads above recreated profile, as the app
    # would, so --force below overwrites it).
    profile.unlink(missing_ok=True)
    runs.unlink(missing_ok=True)
    assert profile_store.get_application(app_id) is None
    # Restore via the script.
    r = _run_script("restore", str(backup_dir), "--force", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    detail = profile_store.get_application(app_id)
    assert detail is not None
    assert detail["resume_text"] == "Resume"
    assert run_store.get("sess-bk1") is not None
    assert db_schema_version(profile) == 1
    assert db_schema_version(runs) == 1
