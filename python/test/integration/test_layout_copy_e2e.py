# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import sys

import numpy as np
import pytest

from aster import ir, utils
from aster.layout import Layout, product
from aster.testing import execute_kernel_and_verify, hsaco_file

# Import shared helpers from python/test/test_layout_codegen.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from test_layout_codegen import build_copy_kernel, compile_to_asm, build_and_compile_copy_kernel

MCPU = "gfx942"
NUM_THREADS = 64
BYTES_PER_THREAD = 16  # dwordx4 = 4 * 4 bytes


@pytest.mark.parametrize(
    "thread_layout",
    [
        # Identity: thread i reads at byte offset i*16
        Layout(sizes=NUM_THREADS, strides=BYTES_PER_THREAD),
        # 2D: 4 threads contiguous (stride 16), 16 groups (stride 64)
        Layout(sizes=(4, 16), strides=(BYTES_PER_THREAD, 4 * BYTES_PER_THREAD)),
    ],
    ids=["identity", "2d_tiled"],
)
def test_layout_copy_e2e(thread_layout):
    """Layout-driven copy kernel: output == input."""
    num_threads = product(thread_layout.sizes)
    total_bytes = num_threads * BYTES_PER_THREAD
    num_floats = total_bytes // 4

    src = np.arange(num_floats, dtype=np.float32)
    dst = np.zeros(num_floats, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        asm = build_and_compile_copy_kernel("copy_e2e", thread_layout)

    path = utils.assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler does not support {MCPU}")

    with hsaco_file(path):
        if not utils.system_has_mcpu(mcpu=MCPU):
            pytest.skip(f"{MCPU} GPU not available")

        execute_kernel_and_verify(
            hsaco_path=path,
            kernel_name="copy_e2e",
            input_args=[src],
            output_args=[dst],
            mcpu=MCPU,
            wavefront_size=64,
            grid_dim=(1, 1, 1),
            block_dim=(num_threads, 1, 1),
        )

    np.testing.assert_array_equal(dst, src)


# ---------------------------------------------------------------------------
# Standalone: python test_layout_copy_e2e.py  -- prints IR + pipeline + ASM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    layout = Layout(sizes=(4, 16), strides=(BYTES_PER_THREAD, 4 * BYTES_PER_THREAD))

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = build_copy_kernel("copy_demo", layout)
        print(module)
        asm = compile_to_asm(module, print_ir_after_all=True)
        print(asm)
