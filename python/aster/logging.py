"""Logging utilities for aster."""

import os
import logging

__all__ = [
    "MillisecondFormatter",
    "_get_logger",
    "_should_log",
    "_log_info",
    "_log_with_device",
]


class MillisecondFormatter(logging.Formatter):
    """Formatter that includes milliseconds in timestamps."""

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        msecs = int(record.msecs)
        return f"{ct.tm_hour:02d}:{ct.tm_min:02d}:{ct.tm_sec:02d}.{msecs:03d}"


def _get_logger():
    """Get logger configured for multiprocessing-safe logging."""
    logger = logging.getLogger("aster")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            MillisecondFormatter(fmt="%(asctime)s [PID:%(process)d] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _should_log():
    """Check if logging is enabled via ASTER_LOGGING environment variable."""
    return bool(os.getenv("ASTER_LOGGING"))


def _log_info(logger, message):
    """Log info message if logging is enabled."""
    if not _should_log():
        return
    logger.info(message)


def _log_with_device(logger, device_id, message):
    """Log message with device_id."""
    if not _should_log():
        return
    device_str = f"GPU{device_id}" if device_id is not None else "GPU?"
    logger.info(f"[{device_str}] {message}")
