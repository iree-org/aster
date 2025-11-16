#!/usr/bin/env python3
"""Smoke check script for example kernels.

This script runs basic examples to verify the build is working correctly. Can be run
standalone or via pytest.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def run_example(script_path: Path, output_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), "-o", str(output_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Failed to run {script_path.name}:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert os.path.exists(output_path), f"Output file {output_path} was not created"


@pytest.fixture
def output_dir(tmp_path):
    """Pytest fixture providing a temporary output directory."""
    output = tmp_path / "amdgcn_smoke"
    output.mkdir(parents=True, exist_ok=True)
    return output


def test_vmov_immediate(output_dir):
    """Test vmov immediate instruction example."""
    script_dir = Path(__file__).parent
    script_path = script_dir / "ex_01_cdna3_vmov_imm_32b.py"
    output_path = output_dir / "smoke_vmov"
    run_example(script_path, output_path, ["--mcpu", "gfx942"])


def test_matmul_v1(output_dir):
    """Test matmul v1 kernel."""
    script_dir = Path(__file__).parent
    script_path = script_dir / "ex_10_cdna3_matmul_v1.py"
    output_path = output_dir / "smoke_matmul_v1"
    run_example(script_path, output_path, ["--mcpu", "gfx942", "--num-vgprs=120"])


def test_matmul_v3(output_dir):
    """Test matmul v3 kernel (basic configuration)."""
    script_dir = Path(__file__).parent
    script_path = script_dir / "ex_10_cdna3_matmul_v3.py"
    output_path = output_dir / "smoke_matmul_v3"
    run_example(script_path, output_path, ["--mcpu", "gfx942", "--num-vgprs=120"])


def test_matmul_v3_agpr(output_dir):
    """Test matmul v3 kernel with AGPRs."""
    script_dir = Path(__file__).parent
    script_path = script_dir / "ex_10_cdna3_matmul_v3.py"
    output_path = output_dir / "smoke_matmul_v3_agpr"
    run_example(
        script_path,
        output_path,
        ["--mcpu", "gfx942", "--num-vgprs=120", "--num-agprs=200"],
    )
