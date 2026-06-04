# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for coordinate-based bounds checks on buffer voffsets.
# The row-major tail is handled by num_records.
# Inner dimensions are checked with coord comparisons.

import math

from aster import ir
from aster.dialects.amdgcn import AccessKind
from aster.dialects.kernel_builder_with_layouts import (
    KernelBuilderWithLayouts,
)
from aster.layout import CoordTensor, Layout, LogicalTensor, Symbol, Tensor, tile

F32 = 4


def _builder():
    b = KernelBuilderWithLayouts("m", "k", target="gfx942")
    b.set_grid_dims(1)
    b.add_ptr_arg(AccessKind.ReadOnly)
    (ptr,) = b.load_args()
    return b, ptr


def test_coord_tensor_identity_projections():
    row, col = Symbol("row"), Symbol("col")
    proj = CoordTensor.tiled((5, 8), (1, 4), axes=(row, col)).projections
    # tile() lays out (inter..., intra...): (row, coltile, row-intra, col-intra).
    assert proj[row].flat_sizes == (5, 2, 1, 4)
    assert proj[row].strides == (1, 0, 1, 0)  # row projection: unit on row only
    assert proj[col].strides == (0, 4, 0, 1)  # col projection: coltile * 4 + intra * 1


test_coord_tensor_identity_projections()


def test_coord_carry_emits_inner_elem_less():
    """Per-dimension bounds emit one select for the inner dimension in 2-D."""
    M, N, TILE_N = 5, 7, 8
    padded_n = math.ceil(N / TILE_N) * TILE_N
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        b, ptr = _builder()
        row, col = Symbol("row"), Symbol("col")
        tiled = tile(Layout((M, padded_n)), (1, TILE_N), axes=(row, col))
        coords = CoordTensor.tiled((M, padded_n), (1, TILE_N), axes=(row, col))
        elem = Layout((M, N), (N * F32, F32))
        ten = LogicalTensor(Tensor(ptr, layout=tiled), elem_layout=elem, coord=coords)
        buf = b.prepare_transfer_tiles_buffer(ptr, buffer_num_records_bytes=M * N * F32)
        sub = b.slice(ten, {row: b.constant_index(0), col: b.constant_index(0)})
        sub.buffer_voffset(b, buf, b.constant_index(0))
        text = str(b._outer)
        # Exactly one select: the inner (column) check; the row tail is dropped by
        # num_records, not predicated.
        assert text.count("arith.select") == 1, text
        # Checked against the LOGICAL extent N, never the padded iteration grid.
        assert f" {N} : index" in text, (
            f"expected a bound against logical N={N}:\n{text}"
        )
        assert f" {padded_n} : index" not in text, (
            "must not compare against padded grid"
        )


def test_no_coord_tensor_no_per_dim_select():
    """Without coord, buffer path uses only the linear num_records bound."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        b, ptr = _builder()
        buf = b.prepare_transfer_tiles_buffer(ptr, buffer_num_records_bytes=140)
        ten = LogicalTensor(
            Tensor(ptr, layout=Layout((35,), (1,))), elem_layout=Layout((5, 7), (8, 1))
        )
        ten.buffer_voffset(b, buf, b.constant_index(0))
        assert "arith.select" not in str(b._outer)
