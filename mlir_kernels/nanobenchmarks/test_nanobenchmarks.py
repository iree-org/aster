"""Pytest wrapper to run all nanobenchmarks with default arguments."""

import runpy
import sys
from unittest.mock import patch
from pathlib import Path

import pytest

# Discover all nanobench_*.py files in this directory
_NANOBENCH_DIR = Path(__file__).parent
_NANOBENCH_MODULES = sorted(p.stem for p in _NANOBENCH_DIR.glob("nanobench_*.py"))


@pytest.mark.parametrize("module_name", _NANOBENCH_MODULES)
def test_nanobench(module_name: str):
    # We need access to utils.py that is not shipped with the aster package.
    nanobench_dir = str(_NANOBENCH_DIR)
    if nanobench_dir not in sys.path:
        sys.path.insert(0, nanobench_dir)
    with patch.object(sys, "argv", [module_name]):
        runpy.run_path(str(_NANOBENCH_DIR / f"{module_name}.py"), run_name="__main__")
