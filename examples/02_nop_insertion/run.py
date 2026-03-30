#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Example 02: The compiler inserts NOPs where the hardware needs them.

Each thread stores its ID to output[tid], then immediately overwrites v0.
The memory unit is still reading v0 when the overwrite happens -- the
amdgcn-hazards pass inserts v_nop delays to prevent stale data.

  python examples/02_nop_insertion/run.py                       # execute on GPU
  python examples/02_nop_insertion/run.py --print-asm           # also print assembly
  python examples/02_nop_insertion/run.py --print-ir-after-all  # IR after each pass
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import section, compile_file, execute_or_skip, here, parse_args

KERNEL = "kernel"
opts = parse_args()
mlir_file = here("kernel.mlir")

PASS_PIPELINE = "builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-hazards)))"

section("(cross-)compile with insertion of hazard-avoiding NOPs")
with open(mlir_file) as f:
    if opts.print_ir_after_all:
        print(f.read())
asm = compile_file(mlir_file, KERNEL, pipeline=PASS_PIPELINE, print_opts=opts)
if opts.print_asm:
    print(asm)
print("(cross-)compile done")

section("execute on GPU")
output = np.zeros(64, dtype=np.int32)
result = execute_or_skip(asm, KERNEL, outputs=[output])
if result is not None:
    expected = np.arange(64, dtype=np.int32)
    if np.array_equal(output, expected):
        print(f"PASS: output = thread IDs [0..63]")
    else:
        print(f"FAIL: expected [0..63], got {output}")
