"""Pytest wrapper to run all nanobenchmarks with default arguments."""

import importlib.util
import sys
from unittest.mock import patch
from pathlib import Path

import pytest

# Discover all nanobench_*.py files in this directory
_NANOBENCH_DIR = Path(__file__).parent
_NANOBENCH_MODULES = sorted(p.stem for p in _NANOBENCH_DIR.glob("nanobench_*.py"))


def _load_and_run_main(module_name: str):
    """Import module and run its main() function."""
    spec = importlib.util.spec_from_file_location(
        module_name, _NANOBENCH_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch.object(sys, "argv", [module_name]):
        module.main()


@pytest.mark.parametrize("module_name", _NANOBENCH_MODULES)
def test_nanobench(module_name: str):
    _load_and_run_main(module_name)
