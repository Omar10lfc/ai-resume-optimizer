#!/usr/bin/env python
"""Backup & restore for the app's three SQLite stores (Item 11).

Stores backed up (per agent/config.py defaults):
  profile:      data/profile.sqlite3                (application history)
  runs:         data/artifacts/runs.sqlite3          (MR3 runs ledger)
  checkpoints:  data/checkpoints/checkpoints.sqlite3 (LangGraph checkpointer;
                skipped when CHECKPOINTER_DB_PATH is off/empty)

Usage:
  python scripts/backup_restore.py backup [--out DIR]
      (default DIR: backups/backup_<timestamp>)
  python scripts/backup_restore.py restore DIR [--force]

Notes:
  - Backups are PLAINTEXT copies of private data (names/emails/phones in
    resumes, run payloads). Store or transfer them encrypted or with
    restrictive permissions.
  - The backup uses sqlite3's backup API: a read-only, consistent snapshot
    of each live store (safe with the app running).
  - Restore REPLACES the live files - the app must be STOPPED first. The
    script refuses to overwrite an existing destination unless --force is
    passed.
  - After restore, the versioned migration table (_migrations, version 1,
    stamped by agent/helpers.py ensure_db_migrations) is ensured on each
    store, and the restored versions are printed.
"""

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import config as agent_config  # noqa: E402
from agent.helpers import db_schema_version, ensure_db_migrations  # noqa: E402


def _stores() -> list[tuple[str, Path]]:
    """The stores to back up, as (name, path). Skips the stores that are
    disabled: PROFILE_DB_PATH=off (PROFILE_DB_ENABLED False) and
    CHECKPOINTER_DB_PATH off/none/memory (empty)."""
    stores: list[tuple[str, Path]] = []
    if agent_config.PROFILE_DB_ENABLED:
        stores.append(("profile", Path(agent_config.PROFILE_DB_PATH)))
    stores.append(("runs", Path(agent_config.RUNS_DB_PATH)))
    ckpt = (agent_config.CHECKPOINTER_DB_PATH or "").strip().lower()
    if ckpt not in ("", "off", "none", "memory"):
        stores.append(("checkpoints", Path(agent_config.CHECKPOINTER_DB_PATH)))
    return stores


def backup(out_dir: Path | None) -> int:
    """Writes a backup of the three stores to out_dir + manifest.json."""
    out_dir = out_dir or (
        REPO_ROOT / "backups" / f"backup_{time.strftime('%Y%m%dT%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "stores": {}}
    for name, path in _stores():
        if not path.exists():
            print(f"  skip (no file): {path}")
            continue
        dest = out_dir / f"{name}.sqlite3"
        dest.unlink(missing_ok=True)
        src = sqlite3.connect(str(path))
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()
        manifest["stores"][name] = {
            "backup": str(dest),
            "path": str(path),
            "schema_version": db_schema_version(dest),
        }
        print(f"  {name}: {path} -> {dest} "
              f"(schema_version={manifest['stores'][name]['schema_version']})")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Backup written to {out_dir}")
    return 0


def restore(backup_dir: Path, force: bool) -> int:
    """Restores the stores from a backup dir. Requires --force when the
    destination files exist (the app must be stopped; restore replaces
    live files)."""
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No manifest.json in {backup_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, info in sorted(manifest["stores"].items()):
        src = backup_dir / info["backup"]
        dest = Path(info["path"])
        if not src.exists():
            print(f"  missing in backup (skipped): {src}")
            continue
        if dest.exists() and not force:
            raise SystemExit(
                f"{dest} exists. Stop the app and restore over it, or pass "
                "--force to overwrite live data.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            stale = Path(str(dest) + suffix)
            if stale.exists():
                try:
                    stale.unlink()
                except OSError as exc:
                    print(f"  warn: could not remove {stale} ({exc}); skipped.")
        try:
            shutil.copyfile(src, dest)
        except OSError as exc:
            print(f"  warn: restore of {name} from {src} failed ({exc}); skipped.")
            continue
        ensure_db_migrations(dest, 1)
        print(f"  {name}: restored {dest} "
              f"(schema_version={db_schema_version(dest)})")
    print(f"Restore complete from {backup_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Backup & restore the app's three SQLite stores (Item 11).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup", help="Back up the stores to a directory.")
    b.add_argument("--out", type=Path, default=None,
                   help="Backup directory (default: backups/backup_<ts>).")
    r = sub.add_parser("restore", help="Restore stores from a backup directory.")
    r.add_argument("dir", type=Path, help="Backup directory (from backup).")
    r.add_argument("--force", action="store_true",
                   help="Overwrite existing destination files. Required when "
                        "the destination exists; always stop the app first.")
    args = ap.parse_args()
    if args.cmd == "backup":
        return backup(args.out)
    return restore(args.dir, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
