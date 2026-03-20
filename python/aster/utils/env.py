"""Environment variable and logging helpers."""

import logging
import os
from typing import Any, Optional


class MillisecondFormatter(logging.Formatter):
    """Formatter that includes milliseconds in timestamps."""

    def formatTime(self, record, datefmt=None):
        """Format time with millisecond precision."""
        ct = self.converter(record.created)
        msecs = int(record.msecs)
        return f"{ct.tm_hour:02d}:{ct.tm_min:02d}:{ct.tm_sec:02d}.{msecs:03d}"


def aster_parse_bool_env(value: str) -> bool:
    """Parse a boolean-like environment variable string.

    Accepts '1', 'true', 'on' as True and '0', 'false', 'off' as False (case-
    insensitive). Raises ValueError for unrecognized values.
    """
    normalized = value.strip().lower()
    if normalized in ("1", "true", "on"):
        return True
    if normalized in ("0", "false", "off"):
        return False
    raise ValueError(
        f"invalid boolean environment variable value: {value!r}. "
        "Expected one of: 1, true, on, 0, false, off."
    )


def aster_get_env_or_default(name: str, default: Optional[Any] = None) -> Any:
    """Return the value of environment variable *name*, or *default* if unset.

    When *default* is a bool the raw string is parsed with
    :func:`aster_parse_bool_env`. Otherwise the raw string value is returned
    as-is.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return aster_parse_bool_env(raw)
    return raw


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
