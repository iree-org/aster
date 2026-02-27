"""Pytest wrapper to run all benchmarks as smoke tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Discover all benchmark_*.py files in this directory (excluding benchmark_utils.py)
_BENCHMARK_DIR = Path(__file__).parent
_BENCHMARK_MODULES = sorted(
    p.stem for p in _BENCHMARK_DIR.glob("benchmark_*.py") if p.stem != "benchmark_utils"
)

@pytest.mark.parametrize("module_name", _BENCHMARK_MODULES)
def test_benchmark(module_name: str):
    """Run benchmark as subprocess with --smoke-test for quick validation."""
    # We need access to benchmark_utils.py that is not shipped with the aster package.
    script_path = _BENCHMARK_DIR / f"{module_name}.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, str(script_path), "--smoke-test"],
        cwd=_BENCHMARK_DIR.parent.parent,  # Run from project root
        capture_output=True,
        text=True,
        env=env,
    )
    # Print output for visibility
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    # Check for failure
    if result.returncode != 0:
        pytest.fail(
            f"Benchmark {module_name} failed with exit code {result.returncode}"
        )
