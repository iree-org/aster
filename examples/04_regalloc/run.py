#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Example 04: Register allocation -- virtual registers, compiler assigns.

Same load-add-store pattern as 03, but with VIRTUAL registers.
No hand-allocated register numbers -- the compiler's graph-coloring
allocator assigns v0, v1, s[0:1], etc. automatically.

  python examples/04_regalloc/run.py                       # execute on GPU
  python examples/04_regalloc/run.py --print-asm           # see physical registers
  python examples/04_regalloc/run.py --print-ir-after-all  # IR after each pass
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import section, compile_file, execute_or_skip, here, parse_args

KERNEL = "kernel"
opts = parse_args()
mlir_file = here("kernel.mlir")

# amdgcn-backend runs at builtin.module scope (regalloc, wait lowering, CF legalization).
# amdgcn-hazards runs inside amdgcn.kernel scope (NOP insertion needs physical registers).
PASS_PIPELINE = (
    "builtin.module("
    "  amdgcn-backend,"
    "  amdgcn.module(amdgcn.kernel(amdgcn-hazards))"
    ")"
)

section("(cross-)compile with register allocation")
with open(mlir_file) as f:
    if opts.print_ir_after_all:
        print(f.read())
asm = compile_file(mlir_file, KERNEL, pipeline=PASS_PIPELINE, print_opts=opts)
if opts.print_asm:
    print(asm)
print("(cross-)compile done")

section("execute on GPU")
N = 64
inp = np.arange(N, dtype=np.int32)
out = np.zeros(N, dtype=np.int32)
result = execute_or_skip(asm, KERNEL, inputs=[inp], outputs=[out])
if result is not None:
    expected = inp + 42
    if np.array_equal(out, expected):
        print(f"PASS: out = in + 42 (first 8: {out[:8]})")
    else:
        print(f"FAIL: expected {expected[:8]}, got {out[:8]}")
