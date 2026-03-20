"""Logging helpers for ASTER."""

import logging

from aster.utils.env import aster_get_env_or_default


class MillisecondFormatter(logging.Formatter):
    """Formatter that includes milliseconds in timestamps."""

    def formatTime(self, record, datefmt=None):
        """Format time with millisecond precision."""
        ct = self.converter(record.created)
        msecs = int(record.msecs)
        return f"{ct.tm_hour:02d}:{ct.tm_min:02d}:{ct.tm_sec:02d}.{msecs:03d}"


def aster_get_logger():
    """Get logger configured for multiprocessing-safe logging."""
    logger = logging.getLogger("benchmark")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            MillisecondFormatter(
                fmt="%(asctime)s [PID:%(process)d] %(message)s",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def aster_should_log():
    """Check if logging is enabled via ASTER_LOGGING environment variable."""
    return aster_get_env_or_default("ASTER_LOGGING", False)


def aster_log_with_device(logger, device_id, message):
    """Log message with device_id."""
    if not aster_should_log():
        return
    device_str = f"GPU{device_id}" if device_id is not None else "GPU?"
    logger.info(f"[{device_str}] {message}")


def aster_log_info(logger, message):
    """Log info message if logging is enabled."""
    if not aster_should_log():
        return
    logger.info(message)
