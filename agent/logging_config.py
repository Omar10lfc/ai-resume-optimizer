"""Central logging configuration with levels (Item 10).

Pipeline nodes log through ``agent.helpers._safe_print`` -> ``logger.info``;
with the root logger at the default WARNING level these are invisible.
``setup_logging()`` is called at app startup; the level comes from the
``LOG_LEVEL`` env var (default ``WARNING``).

Log-site policy: resume/job text never lands in logs - sensitive prints are
redacted or dropped at the sites (scanner/reviewer/ats_check/improper/
polisher/cover_letter/provider/pdf_exporter/latex_render/helpers).
"""

import logging
import os


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure the root logger at ``level`` or ``LOG_LEVEL`` (default
    WARNING). Returns the root logger."""
    lvl = (level or os.environ.get("LOG_LEVEL") or "WARNING").strip().upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    return logging.getLogger()
