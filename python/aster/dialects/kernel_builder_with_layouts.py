# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""KernelBuilder extended with layout-first tile operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder

if TYPE_CHECKING:
    from aster.layout import Layout, Swizzle


class KernelBuilderWithLayouts(KernelBuilder):
    """KernelBuilder with layout-first tile operations."""

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
        load_fn=None,
    ) -> list[tuple[ir.Value, ir.Value]]:
        """Load a multi-MFMA tile from global memory.

        Iterates over tile_layout.size() tiles, each loaded at a byte
        offset determined by tile_layout. load_fn(addr) -> (data, tok)
        is the load inst [global_load_dwordx4]. Returns a list of (data,
        token) pairs, one per tile.
        """
        if load_fn is None:
            load_fn = self.global_load_dwordx4

        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        n_subs = tile_layout.size()
        results = []
        thread_off = self.linearize_layout(self.lane_id(), multi_tile_layout)
        for s in range(n_subs):
            # TODO: this is currently python-constexpr, can be pushed down to IR.
            sub_off = tile_layout(s)
            byte_off = self.affine_apply(
                d0 + d1 + sub_off, [tile_byte_offset, thread_off]
            )
            results.append(load_fn(self.global_addr(ptr, byte_off)))
        return results

    def write_multi_tile_to_lds(
        self,
        data: ir.Value,
        lds_base: ir.Value,
        multi_tile_layout: Layout,
        swizzle: Swizzle,
        tile_layout: Layout,
        write_fn=None,
    ) -> list[ir.Value]:
        """Write a tile to LDS via multiple atomic writes with swizzled addressing.

        write_fn(addr) -> (data, tok) is the write inst [ds_write_b64].
        Returns a list of all write tokens.
        """
        if write_fn is None:
            write_fn = self.ds_write_b64

        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        n_subs = tile_layout.size()
        parts = self.split_vx4(data)
        tokens = []
        lane = self.lane_id()
        for s in range(n_subs):
            # TODO: this is currently python-constexpr, can be pushed down to IR.
            sub_off = tile_layout(s)
            tile_off = self.linearize_layout(lane, multi_tile_layout)
            tile_off = self.affine_apply(d0 + sub_off, [tile_off])
            addr = self.index_to_vgpr(
                self.affine_apply(
                    d0 + d1, [lds_base, self.apply_swizzle(tile_off, swizzle)]
                )
            )
            tokens.append(write_fn(parts[s], addr))
        return tokens

    def read_multi_fragment_from_lds(
        self,
        lds_base: ir.Value,
        multi_tile_layout: Layout,
        swizzle: Swizzle,
        tile_layout: Layout,
        read_fn=None,
    ) -> list[tuple[ir.Value, ir.Value]]:
        """Read all MFMA fragments from LDS for a multi-MFMA tile.

        Iterates over tile_layout.size() tiles. For each tile, computes
        lane offset via multi_tile_layout, adds tile_layout(s) byte
        offset, applies swizzle, and reads via read_fn. read_fn(addr) ->
        (data, tok) defaults to ds_read_b64. Returns a list of (data,
        tok) pairs, one per tile.
        """
        if read_fn is None:
            read_fn = self.ds_read_b64

        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        n_subs = tile_layout.size()
        results = []
        lane = self.lane_id()
        for s in range(n_subs):
            # TODO: this is currently python-constexpr, can be pushed down to IR.
            sub_off = tile_layout(s)
            frag_off = self.linearize_layout(lane, multi_tile_layout)
            frag_off = self.affine_apply(d0 + sub_off, [frag_off])
            swizzled = self.apply_swizzle(frag_off, swizzle)
            addr = self.index_to_vgpr(self.affine_apply(d0 + d1, [lds_base, swizzled]))
            results.append(read_fn(addr))
        return results

    def store_multi_fragment_to_global(
        self,
        acc: ir.Value,
        ptr: ir.Value,
        tile_byte_offset: ir.Value,
        multi_tile_layout: Layout,
        tile_layout: Layout,
        store_fn=None,
    ) -> list[ir.Value]:
        """Store an MFMA C accumulator, iterating over tile_layout fragments.

        store_fn(data, addr) -> tok is the store inst
        [global_store_dword]. Returns a list of all store tokens.
        """
        if store_fn is None:
            store_fn = self.global_store_dword

        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        n_subs = tile_layout.size()
        lane = self.lane_id()
        agprs = self.split_ax4(acc)
        n_agprs = len(agprs) // n_subs
        tokens = []
        for s in range(n_subs):
            # TODO: this is currently python-constexpr, can be pushed down to IR.
            sub_off = tile_layout(s)
            for i in range(n_agprs):
                coord = self.affine_apply(d0 * n_agprs + i, [lane])
                off = self.linearize_layout(coord, multi_tile_layout)
                total = self.affine_apply(d0 + d1 + sub_off, [tile_byte_offset, off])
                tokens.append(
                    store_fn(agprs[s * n_agprs + i], self.global_addr(ptr, total))
                )
        return tokens
