"""Workaround for multiproc deadlocks on bespoke configs."""

import os
import sys

_EXITSTATUS = 0


def pytest_sessionfinish(session, exitstatus):
    global _EXITSTATUS
    _EXITSTATUS = int(exitstatus)


def pytest_unconfigure(config):
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(_EXITSTATUS)
