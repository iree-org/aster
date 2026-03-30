# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared utilities for ASTER examples.

Thin wrappers around aster APIs.
"""

import argparse
import os
import inspect
import sys
import sysconfig


from aster import ir
from aster.compiler.core import (
    compile_mlir_file_to_asm,
    compile_mlir_module_to_asm,
    assemble_to_hsaco,
    PrintOptions,
)
from aster.execution.helpers import hsaco_file  # noqa: F401 (used by examples)
from aster.execution.utils import system_has_mcpu
from aster.test_pass_pipelines import (
    TEST_EMPTY_PASS_PIPELINE,
    TEST_SROA_PASS_PIPELINE,
    TEST_SYNCHRONOUS_PASS_PIPELINE,
)

TARGET = "gfx942"
WAVEFRONT = 64


def parse_args():
    """Parse common CLI flags for examples.

    Returns PrintOptions.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--print-asm",
        action="store_true",
        help="Print assembly via compiler diagnostics",
    )
    p.add_argument(
        "--print-ir-after-all",
        action="store_true",
        help="Print IR after each compiler pass",
    )
    args, _ = p.parse_known_args()
    return PrintOptions.from_flags(
        print_asm=args.print_asm,
        print_ir_after_all=args.print_ir_after_all,
    )


def section(title):
    """Print a section header."""
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n")


def here(filename=""):
    """Return absolute path relative to the caller's file directory."""
    caller_dir = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    return os.path.join(caller_dir, filename) if filename else caller_dir


def compile_file(
    mlir_file,
    kernel,
    pipeline=TEST_EMPTY_PASS_PIPELINE,
    library_paths=None,
    preprocess=None,
    print_opts=None,
):
    """Compile MLIR file to assembly string."""
    with ir.Context() as ctx:
        asm, _ = compile_mlir_file_to_asm(
            mlir_file,
            kernel,
            pipeline,
            ctx,
            library_paths=library_paths,
            preprocess=preprocess,
            print_opts=print_opts,
        )
    return asm


def compile_module(
    module, kernel=None, pipeline=None, library_paths=None, print_opts=None
):
    """Compile a programmatic MLIR module to assembly string."""
    return compile_mlir_module_to_asm(
        module,
        pass_pipeline=pipeline,
        library_paths=library_paths,
        kernel_name=kernel,
        print_opts=print_opts,
    )


def execute_or_skip(
    asm, kernel, inputs=None, outputs=None, grid=(1, 1, 1), block=(WAVEFRONT, 1, 1)
):
    """Assemble to HSACO and execute on GPU in a subprocess.

    ROCm's HIP runtime links LLVM symbols that conflict with ASTER's LLVM, so GPU
    execution must happen in a separate process with LD_LIBRARY_PATH set at startup
    (glibc caches it). Compilation (MLIR -> ASM -> HSACO) stays in the main process.
    """
    import json
    import subprocess
    import tempfile
    import numpy as np

    hsaco_path = assemble_to_hsaco(asm, target=TARGET, wavefront_size=WAVEFRONT)
    if hsaco_path is None:
        print("ERROR: Failed to assemble to HSACO")
        return None

    if not system_has_mcpu(mcpu=TARGET):
        print(f"No {TARGET} GPU -- skipping execution (cross-compilation succeeded)")
        return None

    # Save numpy arrays to temp files for the subprocess
    input_files = []
    for a in inputs or []:
        f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(f, a)
        f.close()
        input_files.append(f.name)

    output_files = []
    for a in outputs or []:
        f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(f, a)
        f.close()
        output_files.append(f.name)

    config = {
        "hsaco_path": hsaco_path,
        "kernel_name": kernel,
        "grid": list(grid),
        "block": list(block),
        "input_files": input_files,
        "output_files": output_files,
    }

    # Set LD_LIBRARY_PATH for subprocess so HIP .so files are findable
    env = os.environ.copy()
    rocm_lib = os.path.join(sysconfig.get_path("purelib"), "_rocm_sdk_core", "lib")
    if os.path.isdir(rocm_lib):
        env["LD_LIBRARY_PATH"] = rocm_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    runner = os.path.join(os.path.dirname(__file__), "_gpu_runner.py")
    result = subprocess.run(
        [sys.executable, runner, json.dumps(config)],
        env=env,
        capture_output=True,
        text=True,
    )

    # Clean up input temps
    for f in input_files:
        os.unlink(f)

    if result.returncode != 0:
        print(f"GPU execution failed:\n{result.stderr.strip()}")
        for f in output_files:
            os.unlink(f)
        return None

    # Read back modified output arrays
    for i, f in enumerate(output_files):
        outputs[i][:] = np.load(f)
        os.unlink(f)

    data = json.loads(result.stdout)
    times = data["times"]
    print(f"Execution times (ns): {times}")
    return times
