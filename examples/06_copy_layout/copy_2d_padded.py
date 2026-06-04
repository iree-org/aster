# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from aster import ir
from aster.dialects.amdgcn import AccessKind
from aster.dialects.kernel_builder_with_layouts import (
    KernelBuilderWithLayouts,
    buffer_load_dwordx4,
    buffer_store_dwordx4,
)
from aster.layout import CoordTensor, Layout, LogicalTensor, Symbol, Tensor, tile
from aster.compiler.core import compile_mlir_module_to_asm, PrintOptions
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig
from common import section, execute_or_skip

KERNEL = "copy_2d_padded"
F32 = 4
LANES = 64
DWX4 = 4  # f32 per lane (buffer_load_dwordx4); N must be a multiple of DWX4
TILE_N = LANES * DWX4  # 256 columns per column-tile
SENTINEL = -999.0


def build_copy_2d_padded(M, N, *, depth, num_cu, target="gfx942"):
    """Dense M x N row-major f32 copy, tiled (1 row x TILE_N cols) per wave,
    with genuine per-dimension OOB derived from the (M, N) logical extents."""
    assert N % DWX4 == 0, "N must be a multiple of 4 (dwordx4 must not straddle row N)"
    n_col_tiles = math.ceil(N / TILE_N)
    padded_n = n_col_tiles * TILE_N

    b = KernelBuilderWithLayouts(f"{KERNEL}_mod", KERNEL, target=target)
    b.set_grid_dims(num_cu)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    src_ptr, dst_ptr = b.load_args()

    row, coltile = Symbol("row"), Symbol("col")
    iter_shape = (M, padded_n)
    tile_shape = (1, TILE_N)
    tiled = tile(Layout(iter_shape), tile_sizes=tile_shape, axes=(row, coltile))
    coords = CoordTensor.tiled(iter_shape, tile_shape, axes=(row, coltile))
    elem = Layout((M, N), (N * F32, F32))
    src_t = LogicalTensor(Tensor(src_ptr, layout=tiled), elem_layout=elem, coord=coords)
    dst_t = LogicalTensor(Tensor(dst_ptr, layout=tiled), elem_layout=elem, coord=coords)

    tc_load = b.make_tiled_copy_descriptor(
        buffer_load_dwordx4,
        thread_layout=Layout(LANES, DWX4),
        value_layout=Layout(1, 0),
    )
    tc_store = b.make_tiled_copy_descriptor(
        buffer_store_dwordx4,
        thread_layout=Layout(LANES, DWX4),
        value_layout=Layout(1, 0),
    )

    nbytes = M * N * F32
    src_buf = b.prepare_transfer_tiles_buffer(src_ptr, buffer_num_records_bytes=nbytes)
    dst_buf = b.prepare_transfer_tiles_buffer(dst_ptr, buffer_num_records_bytes=nbytes)

    wg = b.linear_block_id()
    n_work = M * n_col_tiles
    n_batches = max(depth + 1, math.ceil(n_work / num_cu))
    lb, ub, step = b.constant_index(0), b.constant_index(n_batches), b.constant_index(1)
    d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)

    @b.loop(lb, ub, step)
    def _(k):
        with b.stage(0):
            wkid = b.affine_apply(d0 + d1 * num_cu, [wg, k])
            r = b.affine_apply(ir.AffineExpr.get_floor_div(d0, n_col_tiles), [wkid])
            ct = b.affine_apply(ir.AffineExpr.get_mod(d0, n_col_tiles), [wkid])
            loaded = b.transfer_tile(
                b.slice(src_t, {row: r, coltile: ct}), tc_load, buffer=src_buf
            )
        with b.stage(depth):
            wkid = b.affine_apply(d0 + d1 * num_cu, [wg, k])
            r = b.affine_apply(ir.AffineExpr.get_floor_div(d0, n_col_tiles), [wkid])
            ct = b.affine_apply(ir.AffineExpr.get_mod(d0, n_col_tiles), [wkid])
            b.wait_deps(loaded)
            b.transfer_tile(
                b.slice(dst_t, {row: r, coltile: ct}),
                tc_store,
                data=loaded.payloads,
                buffer=dst_buf,
            )

    return b.build()


def run_copy_2d_padded(
    *,
    m: int = 64,
    n: int = 300,
    depth: int = 2,
    num_cu: int = 304,
    print_opts: PrintOptions | None = None,
) -> None:
    """Build, compile, and optionally execute the 2D padded copy kernel."""
    if print_opts is None:
        print_opts = PrintOptions()

    section(f"build copy_2d_padded {m}x{n} depth={depth} (genuine per-dim OOB)")
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = build_copy_2d_padded(m, n, depth=depth, num_cu=num_cu)
        module.operation.verify()
        if print_opts.print_ir_after_all:
            print(module)
        cfg = PipelineConfig(
            lcm_unroll=True, unroll_factor_multiplier=depth, ll_sched=1
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(cfg),
            print_opts=print_opts,
        )
        if print_opts.print_asm:
            print(asm)

    section("execute on GPU")
    src = np.arange(m * n, dtype=np.float32).reshape(m, n)
    dst = np.full_like(src, SENTINEL)
    result = execute_or_skip(
        asm,
        KERNEL,
        inputs=[src],
        outputs=[dst],
        grid=(num_cu, 1, 1),
        block=(LANES, 1, 1),
    )
    if result is not None:
        if np.array_equal(src, dst):
            print(
                f"PASS: copy of {m}x{n} (cols >= N dropped per-dim; "
                f"first 8: {dst.ravel()[:8]})"
            )
        else:
            n_bad = int(np.sum(src != dst))
            print(f"FAIL: {n_bad} mismatches (row0 tail got {dst[0, -4:]})")
            sys.exit(1)


def main():
    import argparse

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=300)  # not a multiple of TILE_N
    p.add_argument("--depth", type=int, default=2, help="tiles in flight")
    p.add_argument("--num-cu", type=int, default=304)
    p.add_argument("--print-asm", action="store_true")
    p.add_argument("--print-ir-after-all", action="store_true")
    args, _ = p.parse_known_args()
    run_copy_2d_padded(
        m=args.m,
        n=args.n,
        depth=args.depth,
        num_cu=args.num_cu,
        print_opts=PrintOptions.from_flags(
            print_asm=args.print_asm, print_ir_after_all=args.print_ir_after_all
        ),
    )


if __name__ == "__main__":
    main()
