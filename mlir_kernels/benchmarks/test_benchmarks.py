"""Pytest wrapper to run all benchmarks with default arguments."""

import importlib.util
import os
from pathlib import Path

import pytest

# Discover all benchmark_*.py files in this directory (excluding benchmark_utils.py)
_BENCHMARK_DIR = Path(__file__).parent
_BENCHMARK_MODULES = sorted(
    p.stem for p in _BENCHMARK_DIR.glob("benchmark_*.py") if p.stem != "benchmark_utils"
)


def _load_and_run_main(module_name: str):
    """Import module and run its main() function."""
    spec = importlib.util.spec_from_file_location(
        module_name, _BENCHMARK_DIR / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


@pytest.mark.parametrize("module_name", _BENCHMARK_MODULES)
def test_benchmark(module_name: str):
    _load_and_run_main(module_name)
