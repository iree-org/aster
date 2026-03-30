#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Example 01: Hand-written AMDGCN MLIR compiled to assembly and run on GPU.

Demonstrates WYSIWYG -- assembly is a direct translation of the MLIR.
Self-check: computes 32 + 10 = 42 and traps if the result is wrong.

  python examples/01_hello_asm/run.py                       # execute on GPU
  python examples/01_hello_asm/run.py --print-asm           # also print assembly
  python examples/01_hello_asm/run.py --print-ir-after-all  # IR after each pass
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import section, compile_file, execute_or_skip, here, parse_args

KERNEL = "kernel"
opts = parse_args()
mlir_file = here("kernel.mlir")

section("(cross-)compile")
asm = ""
with open(mlir_file) as f:
    if opts.print_ir_after_all:
        print(f.read())
    asm = compile_file(mlir_file, KERNEL, print_opts=opts)
    if opts.print_asm:
        print(asm)
print("(cross-)compile done")

section("execute on GPU")
asm = compile_file(mlir_file, KERNEL, print_opts=opts)
result = execute_or_skip(asm, KERNEL)
if result is not None:
    print("PASS: kernel completed without trap (32 + 10 == 42)")
