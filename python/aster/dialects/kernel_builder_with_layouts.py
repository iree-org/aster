# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""KernelBuilder extended with layout-first tile operations."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder

if TYPE_CHECKING:
    from aster.layout import Layout, Swizzle


class WGShuffleStrategy(Enum):
    """Workgroup-to-tile mapping strategy for L2 cache reuse."""

    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    N_SWIZZLE = "n_swizzle"


class KernelBuilderWithLayouts(KernelBuilder):
    """KernelBuilder with layout-first tile operations."""

    # -----------------------------------------------------------------------
    # Workgroup shuffle: linear_block_id -> (m_tile, n_tile)
    # -----------------------------------------------------------------------

    def shuffled_tile_coords(
        self,
        grid_m: int,
        grid_n: int,
        strategy: WGShuffleStrategy = WGShuffleStrategy.ROW_MAJOR,
    ) -> tuple[ir.Value, ir.Value]:
        """Emit MLIR ops: linear_block_id -> (m_tile_idx, n_tile_idx)."""
        from aster.layout import Layout

        def _max_admissible_log(n: int) -> int:
            for k in range(3, 0, -1):
                if n % (1 << k) == 0:
                    return k
            return 0

        bid = self.linear_block_id()

        if strategy == WGShuffleStrategy.ROW_MAJOR:
            m_layout = Layout((grid_m, grid_n), (1, 0))
            n_layout = Layout((grid_m, grid_n), (0, 1))
        elif strategy == WGShuffleStrategy.COL_MAJOR:
            m_layout = Layout((grid_n, grid_m), (0, 1))
            n_layout = Layout((grid_n, grid_m), (1, 0))
        elif strategy == WGShuffleStrategy.N_SWIZZLE:
            s = 1 << _max_admissible_log(grid_n)
            n_blocks = grid_n // s
            m_layout = Layout((s, n_blocks, grid_m), (0, 0, 1))
            n_layout = Layout((s, n_blocks, grid_m), (1, s, 0))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        m_idx = self.linearize_layout(bid, m_layout)
        n_idx = self.linearize_layout(bid, n_layout)
        return (m_idx, n_idx)

    # -----------------------------------------------------------------------
    # Multi-MFMA tile ops (compose single-tile ops over tile_layout)
    #
    # Each multi op takes:
    #   multi_tile_layout -- per-thread layout within the full multi-MFMA tile
    #   tile_layout       -- byte offset layout such that:
    #     sizes == multiplicity, strides = per-tile byte offsets
    # -----------------------------------------------------------------------

    def load_multi_tile_from_global(
        self,
        ptr: ir.Value,
        tile_byte_offset: ir.Value,
        multi_tile_layout: Layout,
        tile_layout: Layout,
        load_fn,
    ) -> list[tuple[ir.Value, ir.Value]]:
        """Load a multi-MFMA tile from global memory.

        load_fn(addr, dynamic_offset=vgpr) -> (data, tok).

        Returns a list of (data, token) pairs, one per sub-tile.
        """
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        n_subs = tile_layout.size()
        results = []
        thread_off = self.linearize_layout(self.lane_id(), multi_tile_layout)
        for s in range(n_subs):
            sub_off = tile_layout(s)
            byte_off = self.affine_apply(
                d0 + d1 + sub_off, [tile_byte_offset, thread_off]
            )
            results.append(load_fn(ptr, dynamic_offset=self.index_to_vgpr(byte_off)))
        return results

    def write_multi_tile_to_lds(
        self,
        data: ir.Value,
        lds_base: ir.Value,
        multi_tile_layout: Layout,
        swizzle: Swizzle,
        tile_layout: Layout,
        write_fn,
    ) -> list[ir.Value]:
        """Write a tile to LDS with swizzled addressing.

        write_fn(data, addr) -> tok. Returns a list of write tokens.
        """
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        n_subs = tile_layout.size()
        parts = [data] if n_subs == 1 else self.split_vx4(data)
        tokens = []
        lane = self.lane_id()
        for s in range(n_subs):
            sub_off = tile_layout(s)
            tile_off = self.linearize_layout(lane, multi_tile_layout)
            tile_off = self.affine_apply(d0 + sub_off, [tile_off])
            addr = self.affine_apply(
                d0 + d1, [lds_base, self.apply_swizzle(tile_off, swizzle)]
            )
            tokens.append(write_fn(parts[s], addr))
        return tokens

    def read_multi_fragment_from_lds(
        self,
        lds_base: ir.Value,
        multi_tile_layout: Layout,
        swizzle: Swizzle,
        tile_layout: Layout,
        read_fn,
    ) -> list[tuple[ir.Value, ir.Value]]:
        """Read MFMA fragments from LDS with swizzle.

        read_fn(addr) -> (data, tok). Returns a list of (data, tok)
        pairs, one per sub-tile.
        """
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        n_subs = tile_layout.size()
        results = []
        lane = self.lane_id()
        for s in range(n_subs):
            sub_off = tile_layout(s)
            frag_off = self.linearize_layout(lane, multi_tile_layout)
            frag_off = self.affine_apply(d0 + sub_off, [frag_off])
            swizzled = self.apply_swizzle(frag_off, swizzle)
            addr = self.affine_apply(d0 + d1, [lds_base, swizzled])
            results.append(read_fn(addr))
        return results

    def store_multi_fragment_to_global(
        self,
        acc: ir.Value,
        ptr: ir.Value,
        tile_byte_offset: ir.Value,
        multi_tile_layout: Layout,
        tile_layout: Layout,
        store_fn,
    ) -> list[ir.Value]:
        """Store MFMA C accumulator to global memory.

        store_fn(data, addr, dynamic_offset=vgpr) -> tok.

        Returns a list of store tokens.
        """
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        n_subs = tile_layout.size()
        lane = self.lane_id()
        agprs = self.split_ax4(acc)
        n_agprs = len(agprs) // n_subs
        tokens = []
        for s in range(n_subs):
            sub_off = tile_layout(s)
            for i in range(n_agprs):
                coord = self.affine_apply(d0 * n_agprs + i, [lane])
                off = self.linearize_layout(coord, multi_tile_layout)
                total = self.affine_apply(d0 + d1 + sub_off, [tile_byte_offset, off])
                tokens.append(
                    store_fn(
                        agprs[s * n_agprs + i],
                        ptr,
                        dynamic_offset=self.index_to_vgpr(total),
                    )
                )
        return tokens
