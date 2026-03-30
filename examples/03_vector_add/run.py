#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Example 03: Vector addition -- c[tid] = a[tid] + b[tid].

First kernel that loads from memory, computes, and stores back.
Three buffer arguments, two global loads, one add, one store.

  python examples/03_vector_add/run.py                       # execute on GPU
  python examples/03_vector_add/run.py --print-asm           # also print assembly
  python examples/03_vector_add/run.py --print-ir-after-all  # IR after each pass
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import section, compile_file, execute_or_skip, here, parse_args

KERNEL = "kernel"
opts = parse_args()
mlir_file = here("kernel.mlir")

PASS_PIPELINE = (
    "builtin.module("
    "  inline,symbol-dce,"
    "  amdgcn.module(amdgcn.kernel(amdgcn-hazards))"
    ")"
)

section("(cross-)compile")
with open(mlir_file) as f:
    if opts.print_ir_after_all:
        print(f.read())
asm = compile_file(mlir_file, KERNEL, pipeline=PASS_PIPELINE, print_opts=opts)
if opts.print_asm:
    print(asm)
print("(cross-)compile done")

section("execute on GPU")
N = 64
a = np.arange(N, dtype=np.int32)
b = np.full(N, 100, dtype=np.int32)
c = np.zeros(N, dtype=np.int32)
result = execute_or_skip(asm, KERNEL, inputs=[a, b], outputs=[c])
if result is not None:
    expected = a + b
    if np.array_equal(c, expected):
        print(f"PASS: c = a + b (first 8: {c[:8]})")
    else:
        print(f"FAIL: expected {expected[:8]}, got {c[:8]}")
