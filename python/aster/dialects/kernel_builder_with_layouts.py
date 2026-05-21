# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""KernelBuilder extended with layout-first tile operations."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder
from aster.layout.algebra import Layout, Swizzle
from aster.layout.tensor import Tensor

if TYPE_CHECKING:
    pass


class MemSpace(enum.Enum):
    """Address space of a copy endpoint."""

    GLOBAL = "global"
    REG = "reg"
    LDS = "lds"


class Scope(enum.Enum):
    """Participant universe a TiledCopy partitions across."""

    LANE = "lane"
    WAVE = "wave"  # NYI
    THREAD = "thread"  # NYI
    WORKGROUP = "workgroup"  # NYI


@dataclass(frozen=True)
class Copy:
    """A hardware transfer: the ISA op + its endpoints' address spaces."""

    op_name: str
    src_space: MemSpace
    dst_space: MemSpace


global_load_dwordx4 = Copy("global_load_dwordx4", MemSpace.GLOBAL, MemSpace.REG)
ds_write_64b = Copy("ds_write_b64", MemSpace.REG, MemSpace.LDS)
ds_read_64b = Copy("ds_read_b64", MemSpace.LDS, MemSpace.REG)


@dataclass(frozen=True)
class TiledCopy:
    """A copy plan bound to a participant: hardware atomic copy + byte-space
    thread/value layouts + the (pid, scope) the layouts are addressed at."""

    copy: Copy
    thread_layout: Layout
    value_layout: Layout
    pid: Any
    scope: Scope
    swizzle: Optional[Swizzle] = None


class KernelBuilderWithLayouts(KernelBuilder):
    """KernelBuilder with layout-first tile operations."""

    def TensorDescriptor(self, ptr: Any, offset: Any = None) -> Tensor:
        """Build the memory side of a copy: a pointer and an optional dynamic offset."""
        return Tensor(ptr, offset)

    def _scope_count(self, scope: Scope) -> int:
        if scope is Scope.LANE:
            # wave_size on CDNA; matches lane_id()'s default
            # TODO: programmatically extract from target info
            return 64
        raise NotImplementedError(f"scope {scope.value!r} has no count wired")

    def _scope_pid(self, scope: Scope) -> Any:
        if scope is Scope.LANE:
            return self.lane_id()
        raise NotImplementedError(f"scope {scope.value!r} has no pid producer wired")

    def make_tiled_copy_descriptor(
        self,
        copy: Copy,
        thread_layout: Layout,
        value_layout: Layout,
        *,
        swizzle: Optional[Swizzle] = None,
        scope: Scope = Scope.LANE,
    ) -> TiledCopy:
        expected = self._scope_count(scope)
        actual = thread_layout.size()
        assert actual == expected, (
            f"thread_layout size {actual} does not match scope "
            f"{scope.value!r} count {expected}; the participant id chosen "
            "for this scope would index outside the layout."
        )
        return TiledCopy(
            copy=copy,
            thread_layout=thread_layout,
            value_layout=value_layout,
            pid=self._scope_pid(scope),
            scope=scope,
            swizzle=swizzle,
        )

    def _addr(self, base: Any, off: ir.Value, swizzle: Optional[Swizzle]) -> Any:
        """LDS / global address = base + (maybe swizzled) per-transfer offset."""
        rel = off if swizzle is None else self.apply_swizzle(off, swizzle)
        return self.layout_sum(base, rel)

    def copy(
        self,
        tc: TiledCopy,
        tensor: Tensor,
        data: Any = None,
    ):
        """Emit N hardware transfers for one TiledCopy."""
        copy_atom = tc.copy
        swizzle = tc.swizzle

        offsets = self.thread_value_offsets(
            tc.pid,
            tc.thread_layout,
            tc.value_layout,
        )
        n = len(offsets)

        # Fold the tensor's base offset into each per-v offset.
        def _total(voff: ir.Value) -> ir.Value:
            return (
                self.layout_sum(tensor.offset, voff)
                if tensor.offset is not None
                else voff
            )

        out: list[Any] = []
        if (
            copy_atom.src_space is MemSpace.GLOBAL
            and copy_atom.dst_space is MemSpace.REG
        ):
            op = getattr(self, copy_atom.op_name)
            for voff in offsets:
                out.append(
                    op(tensor.ptr, dynamic_offset=self.index_to_vgpr(_total(voff)))
                )
        elif (
            copy_atom.src_space is MemSpace.REG and copy_atom.dst_space is MemSpace.LDS
        ):
            assert data is not None, "reg->lds copy needs data"
            if not isinstance(data, (list, tuple)):
                data = [data]
            assert len(data) == n, (
                f"reg->lds copy: data has {len(data)} payloads but "
                f"value_layout.size()={n}"
            )
            op = getattr(self, copy_atom.op_name)
            for v, voff in enumerate(offsets):
                out.append(op(data[v], self._addr(tensor.ptr, _total(voff), swizzle)))
        elif (
            copy_atom.src_space is MemSpace.LDS and copy_atom.dst_space is MemSpace.REG
        ):
            op = getattr(self, copy_atom.op_name)
            for voff in offsets:
                out.append(op(self._addr(tensor.ptr, _total(voff), swizzle)))
        else:
            raise NotImplementedError(
                f"copy: {copy_atom.src_space} -> {copy_atom.dst_space}"
            )
        return out[0] if n == 1 else out

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
        thread_off = self.layout_apply(self.lane_id(), multi_tile_layout)
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
            tile_off = self.layout_apply(lane, multi_tile_layout)
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
            frag_off = self.layout_apply(lane, multi_tile_layout)
            frag_off = self.affine_apply(d0 + sub_off, [frag_off])
            swizzled = self.apply_swizzle(frag_off, swizzle)
            addr = self.affine_apply(d0 + d1, [lds_base, swizzled])
            results.append(read_fn(addr))
        return results

    # -----------------------------------------------------------------------
    # MFMA C-accumulator store.-----------------------------------------------------------------------

    def store_multi_fragment_to_global(
        self,
        acc: ir.Value,
        ptr: ir.Value,
        tile_byte_offset: ir.Value,
        multi_tile_layout: Layout,
        tile_layout: Layout,
        store_fn,
        nt: bool = True,
    ) -> list[ir.Value]:
        """Store MFMA C accumulator to global memory.

        store_fn(data, addr, dynamic_offset=vgpr, nt=bool) -> tok.

        Default Non-Temporal (nt) bit streams through L2 without alloc
        LRU line.

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
                off = self.layout_apply(coord, multi_tile_layout)
                total = self.affine_apply(d0 + d1 + sub_off, [tile_byte_offset, off])
                tokens.append(
                    store_fn(
                        agprs[s * n_agprs + i],
                        ptr,
                        dynamic_offset=self.index_to_vgpr(total),
                        nt=nt,
                    )
                )
        return tokens
