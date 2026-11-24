"""Regression test for independent module imports in fresh subprocesses.

Guards against missing attributes or circular import failures that can hide
behind pytest runner's module caching and import ordering.
"""
import subprocess
import sys
import pytest


@pytest.mark.parametrize("module_name", [
    "agent",
    "agent.ats",
    "agent.match",
    "server.app",
])
def test_fresh_subprocess_import(module_name: str):
    """Ensure core modules can be imported independently in a fresh subprocess."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"Importing '{module_name}' failed in fresh subprocess with exit code {proc.returncode}:\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
