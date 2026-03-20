"""Environment variable helpers."""

import os
from typing import Any, Optional


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
