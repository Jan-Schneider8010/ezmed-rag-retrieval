"""Logging setup using rich for readable local output."""

import logging
import resource

from rich.logging import RichHandler

from ezmed.settings import settings

logger = logging.getLogger(__name__)

# High --workers runs open hundreds of concurrent sockets. The default macOS
# soft limit (256 open files) exhausts descriptors mid-run — every socket, disk
# cache write, and even tempfile.mkstemp then fails with "Too many open files".
# The hard limit is far higher, so lift the soft limit to match at startup.
_FD_SOFT_TARGET = 65536


def raise_fd_soft_limit() -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return
    target = _FD_SOFT_TARGET if hard == resource.RLIM_INFINITY else min(_FD_SOFT_TARGET, hard)
    if soft >= target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        logger.debug("raised open-file soft limit %d -> %d", soft, target)
    except (ValueError, OSError):
        pass


def configure_logging() -> None:
    raise_fd_soft_limit()
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
