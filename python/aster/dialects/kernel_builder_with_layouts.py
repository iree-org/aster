# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""KernelBuilder extended with layout-first tile operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder

if TYPE_CHECKING:
    from aster.layout import Layout, Swizzle


class KernelBuilderWithLayouts(KernelBuilder):
    """KernelBuilder with layout-first tile operations."""

    # -----------------------------------------------------------------------
    # Single-tile ops
    # -----------------------------------------------------------------------

    def load_tile_from_global(
        self,
        ptr: ir.Value,
        tile_byte_offset: ir.Value,
        tile_layout: Layout,
    ) -> tuple[ir.Value, ir.Value]:
        """Load a tile from global memory using layout for coalescing."""
        thread_off = self.linearize_layout(self.lane_id(), tile_layout)
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        byte_off = self.affine_apply(d0 + d1, [tile_byte_offset, thread_off])
        return self.global_load_dwordx4(self.global_addr(ptr, byte_off))

    # TODO: still too tied to mfma_f32_16x16x16_f16 layout into a dual-tile of 16x16x32, need to generalize.
    def store_fragment_to_global(
        self,
        acc: ir.Value,
        ptr: ir.Value,
        tile_byte_offset: ir.Value,
        fragment_layout: Layout,
    ) -> None:
        """Store an MFMA C accumulator fragment to global memory."""
        lane = self.lane_id()
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        a0, a1, a2, a3 = self.split_ax4(acc)
        for i, agpr in enumerate([a0, a1, a2, a3]):
            coord = self.affine_apply(d0 * 4 + i, [lane])
            off = self.linearize_layout(coord, fragment_layout)
            total = self.affine_apply(d0 + d1, [tile_byte_offset, off])
            self.global_store_dword(agpr, self.global_addr(ptr, total))

    def read_fragment_from_lds(
        self,
        lds_base: ir.Value,
        fragment_layout: Layout,
        swizzle: Swizzle,
        k_byte_offset: Optional[ir.Value] = None,
    ) -> tuple[ir.Value, ir.Value]:
        """Read an MFMA fragment from LDS using fragment layout + swizzle."""
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        fragment_off = self.linearize_layout(self.lane_id(), fragment_layout)
        if k_byte_offset is not None:
            fragment_off = self.affine_apply(d0 + d1, [fragment_off, k_byte_offset])
        swizzled = self.apply_swizzle(fragment_off, swizzle)
        addr = self.index_to_vgpr(self.affine_apply(d0 + d1, [lds_base, swizzled]))
        return self.ds_read_b64(addr)

    # TODO: still too tied to mfma_f32_16x16x16_f16 layout into a dual-tile of 16x16x32, need to generalize.
    def write_tile_to_lds(
        self,
        data: ir.Value,
        lds_base: ir.Value,
        write_layout: Layout,
        swizzle: Swizzle,
    ) -> list[ir.Value]:
        """Write a dwordx4 tile to LDS with swizzled addressing."""
        lane = self.lane_id()
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        tile_off = self.linearize_layout(lane, write_layout)
        tile_off_hi = self.affine_apply(d0 + 8, [tile_off])
        lo_addr = self.index_to_vgpr(
            self.affine_apply(
                d0 + d1, [lds_base, self.apply_swizzle(tile_off, swizzle)]
            )
        )
        hi_addr = self.index_to_vgpr(
            self.affine_apply(
                d0 + d1, [lds_base, self.apply_swizzle(tile_off_hi, swizzle)]
            )
        )
        lo, hi = self.split_vx4(data)
        tok_lo = self.ds_write_b64(lo, lo_addr)
        tok_hi = self.ds_write_b64(hi, hi_addr)
        return [tok_lo, tok_hi]
