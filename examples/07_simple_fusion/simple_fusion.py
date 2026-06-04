#!/usr/bin/env python3
# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Example 07: simple fused add+relu with pinned/unallocated handoff.

python examples/07_simple_fusion/simple_fusion.py                 # execute on GPU (skips if none)
python examples/07_simple_fusion/simple_fusion.py --print-asm     # print the fused assembly
python examples/07_simple_fusion/simple_fusion.py --print-ir-after-all
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from aster import ir
from aster.compiler.core import PrintOptions
from aster.dialects.kernel_builder import KernelBuilder, AccessKind
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE
from common import section, compile_module, execute_or_skip, here, parse_args

KERNEL = "fused"
SIZE = 4
ADD4_FILE = here("add4.mlir")


def _sym_name(op: ir.Operation) -> str:
    return str(op.attributes["sym_name"]).strip('"')


def _import(builder: KernelBuilder, add4_file: str):
    """Load add4 signature from add4.mlir and clone the function into builder."""
    with open(add4_file) as f:
        add4_module = ir.Module.parse(f.read())

    add4_op = None
    for top in add4_module.body.operations:
        if top.operation.name != "amdgcn.module":
            continue
        for inner in top.regions[0].blocks[0].operations:
            if (
                inner.operation.name == "func.func"
                and _sym_name(inner.operation) == "add4"
            ):
                add4_op = inner.operation
                break
        if add4_op is not None:
            break
    if add4_op is None:
        raise ValueError("add4.mlir does not define func.func @add4")

    fn_type = ir.FunctionType(add4_op.attributes["function_type"].value)
    add4_op.clone(ip=ir.InsertionPoint.at_block_begin(builder._mod_block))
    return list(fn_type.inputs), list(fn_type.results)


def build_fused():
    b = KernelBuilder("pin_fusion", KERNEL, target="gfx942")

    add4_arg_types, add4_ret_types = _import(b, ADD4_FILE)

    # relu4(x..., out_ptr): unallocated compute + global writeback.
    relu_input_types = [b.vgpr_type for _ in add4_ret_types] + [b.sgpr2_type]

    @b.define_helper("relu4", relu_input_types, [])
    def relu4(bb, *args):
        xs = list(args[:-1])
        out_ptr = args[-1]
        c0 = bb.constant_i32(0)
        zero = bb.vop1("v_mov_b32", c0)
        max_vals = [bb.vop2("v_max_f32", zero, x) for x in xs]
        data = bb._make_register_range(max_vals)
        # SGPR base + VGPR offset (saddr+vaddr), mirroring add4.mlir's loads.
        off = bb.vop1("v_mov_b32", c0)
        bb._global_store("global_store_dwordx4", data, out_ptr, dynamic_offset=off)
        bb.wait_vmcnt(0)

    # Bridge is synthesized from add4 return types + relu input types.
    bridge_ret_types = relu_input_types[:-1]

    @b.define_helper("bridge_add_to_relu4", add4_ret_types, bridge_ret_types)
    def bridge_add_to_relu4(bb, *s_vals):
        return [bb.vop1("v_mov_b32", s) for s in s_vals]

    # Kernel stays pure Python: load pointers, call add4 -> bridge -> relu4.
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, out_ptr = b.load_args()
    b.set_grid_dims(1, 1, 1)

    s = b.call_helper("add4", [a_ptr, b_ptr], add4_ret_types)
    x = b.call_helper("bridge_add_to_relu4", list(s), bridge_ret_types)
    b.call_helper("relu4", list(x) + [out_ptr], [])

    module = b.build()
    module.operation.verify()
    return module


def run_simple_fusion(*, size: int = SIZE, print_opts: PrintOptions | None = None):
    """Build, compile, and (on a gfx942 GPU) execute the fused add+relu kernel.

    Returns the output array when executed, or None when no GPU is
    present (cross-compilation still runs). Raises AssertionError on a
    value mismatch.
    """
    if print_opts is None:
        print_opts = PrintOptions()

    section("(cross-)compile to a single fused add+relu kernel")
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = build_fused()
        if print_opts.print_ir_after_all:
            print(module)
        asm = compile_module(
            module,
            kernel=KERNEL,
            pipeline=TEST_SROA_PASS_PIPELINE,
            print_opts=print_opts,
        )
        if print_opts.print_asm:
            print(asm)
    print("(cross-)compile done")

    section("execute on GPU")
    a = np.random.randn(size).astype(np.float32)
    b_in = np.random.randn(size).astype(np.float32)
    out = np.zeros(size, dtype=np.float32)
    result = execute_or_skip(
        asm, KERNEL, inputs=[a, b_in], outputs=[out], grid=(1, 1, 1), block=(1, 1, 1)
    )
    if result is not None:
        expected = np.maximum(a + b_in, 0.0)
        assert np.allclose(out, expected, atol=1e-5), (
            f"out != relu(a + b): expected {expected}, got {out}"
        )
        print(f"PASS: out == relu(a + b) (first 4: {out[:4]})")
    return result


def main():
    run_simple_fusion(print_opts=parse_args())


if __name__ == "__main__":
    main()
