#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Example 05: Python Meta-Programming -- generate MLIR from Python, not by hand.

Same out[tid] = in[tid] + 42 as example 04, but the MLIR is generated
programmatically via KernelBuilder instead of written in a .mlir file.

  python examples/05_python/run.py                       # execute on GPU
  python examples/05_python/run.py --print-asm           # also print assembly
  python examples/05_python/run.py --print-ir-after-all  # IR after each pass
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder, AccessKind
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE
from common import section, compile_module, execute_or_skip, parse_args

N = 64
KERNEL = "kernel"
opts = parse_args()


def build_add42():
    """Out[tid] = in[tid] + 42 -- same as 04_regalloc/kernel.mlir."""
    b = KernelBuilder("add42", KERNEL, target="gfx942", isa="cdna3")

    b.add_ptr_arg(AccessKind.ReadOnly)  # in
    b.add_ptr_arg(AccessKind.WriteOnly)  # out
    [in_ptr, out_ptr] = b.load_args()

    # Thread ID -> byte offset (4 bytes per i32)
    tid = b.thread_id("x")
    byte_off = b.affine_apply(ir.AffineExpr.get_dim(0) * 4, [tid])

    # Load, add, store with token-based synchronization
    in_addr = b.global_addr(in_ptr, byte_off)
    val, tok_ld = b.global_load_dword(in_addr)
    b.wait_deps(tok_ld)

    result = b.v_add_u32(b.constant_i32(42), val)

    out_addr = b.global_addr(out_ptr, byte_off)
    tok_st = b.global_store_dword(result, out_addr)
    b.wait_deps(tok_st)

    module = b.build()
    module.operation.verify()
    return module


# -- compile ------------------------------------------------------------------
section("generate MLIR from Python")
with ir.Context() as ctx, ir.Location.unknown():
    module = build_add42()
    if opts.print_ir_after_all:
        print(module)

    section("(cross-)compile")
    asm = compile_module(
        module,
        kernel=KERNEL,
        pipeline=TEST_SROA_PASS_PIPELINE,
        print_opts=opts,
    )
    if opts.print_asm:
        print(asm)
print("(cross-)compile done")

# -- execute ------------------------------------------------------------------
section("execute on GPU")
inp = np.arange(N, dtype=np.int32)
out = np.zeros(N, dtype=np.int32)
result = execute_or_skip(asm, KERNEL, inputs=[inp], outputs=[out])
if result is not None:
    expected = inp + 42
    if np.array_equal(out, expected):
        print(f"PASS: out = in + 42 (first 8: {out[:8]})")
    else:
        print(f"FAIL: expected {expected[:8]}, got {out[:8]}")
