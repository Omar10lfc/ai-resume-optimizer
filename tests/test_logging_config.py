"""Item 10: central logging config with levels (agent/logging_config.py)."""

import logging

from agent.logging_config import setup_logging


def test_setup_logging_default_warning(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    root = setup_logging()
    assert root.level == logging.WARNING
    assert root.handlers  # basicConfig attached a handler


def test_setup_logging_level_from_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    assert setup_logging().level == logging.INFO
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert setup_logging().level == logging.DEBUG
    # An invalid value falls back to WARNING instead of crashing startup.
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_REAL_LEVEL")
    assert setup_logging().level == logging.WARNING
