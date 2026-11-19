"""Extension readiness guards: least-privilege MV3 permissions + store assets."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    return json.loads((ROOT / "extension" / "manifest.json").read_text(
        encoding="utf-8"))


def test_no_blanket_host_permission():
    m = _manifest()
    host = m.get("host_permissions", [])
    assert not any("https://*/*" in p or "http://*/*" in p for p in host), \
        "blanket host access must be opt-in, not granted at install"
    optional = m.get("optional_host_permissions", [])
    assert "https://*/*" in optional
    assert "http://*/*" in optional
    assert any("localhost" in p for p in host), \
        "the default local backend must work without a permission prompt"


def test_content_scripts_keep_job_board_matches():
    m = _manifest()
    matches = [p for rule in m["content_scripts"] for p in rule["matches"]]
    for board in ("linkedin.com", "greenhouse.io", "lever.co", "indeed.com",
                  "myworkdayjobs.com", "ashbyhq.com"):
        assert any(board in p for p in matches), board


def test_cloud_backend_permission_requested_at_runtime():
    src = (ROOT / "frontend" / "src" / "api.ts").read_text(encoding="utf-8")
    assert "ensureBackendHostAccess" in src
    assert "chrome.permissions.request" in src
    assert "await ensureBackendHostAccess(updated.backendUrl)" in src


def test_store_assets_present():
    store = ROOT / "extension" / "store"
    for name in ("listing.md", "privacy_policy.md", "promo-tile.png"):
        assert (store / name).is_file(), name
