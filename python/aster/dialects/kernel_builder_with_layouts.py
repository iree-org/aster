# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""KernelBuilder extended with layout-first tile operations."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias, Union

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder
from aster.layout.algebra import (
    Layout,
    Swizzle,
    Symbol,
    enumerate_flat_coords,
)
from aster.layout.tensor import Tensor
from aster.layout.values import LayoutValues

# Wait arguments to wait_deps:
#   - a LayoutValues with flattened token_values(), or
#   - a list/tuple of tokens, or
#   - a bare ir.Value token.
WaitArg: TypeAlias = Union[LayoutValues, list[ir.Value], tuple[ir.Value, ...], ir.Value]

# Bound values in local_tile.
BoundValue: TypeAlias = Union[int, ir.Value]
# local_tile accepts a mapping from inter-tile-axis Symbol to a value.
Bindings: TypeAlias = dict[Symbol, BoundValue]


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


@dataclass(frozen=True, slots=True)
class _TileTransfer:
    """Per-value payloads and tokens emitted for one tiled transfer."""

    payloads: tuple[Any | None, ...]
    tokens: tuple[Any, ...]


class KernelBuilderWithLayouts(KernelBuilder):
    """KernelBuilder with layout-first tile operations."""

    def wait_deps(self, *tokens: WaitArg) -> None:
        """Wait for dependency tokens (supports various forms of WaitArg)."""

        flat: list[ir.Value] = []
        for tok in tokens:
            if isinstance(tok, LayoutValues):
                flat.extend(tok.token_values())
            elif isinstance(tok, (list, tuple)):
                flat.extend(tok)
            else:
                flat.append(tok)
        super().wait_deps(*flat)

    def _filter_layout_by_axes(
        self, layout: Layout, axes: tuple[Symbol, ...]
    ) -> Layout:
        """Flattened sub-layout of layout restricted to axes."""

        assert layout.axes is not None, (
            f"_filter_layout_by_axes requires axes (from tile()), got {layout!r}"
        )
        flat_sizes = layout.flat_sizes
        flat_strides = (
            layout.strides if isinstance(layout.strides, tuple) else (layout.strides,)
        )
        sizes: list[int] = []
        strides: list[int] = []
        for ax in axes:
            try:
                i = layout.axes.index(ax)
            except ValueError as e:
                raise KeyError(f"axis {ax!r} not in layout axes {layout.axes!r}") from e
            sizes.append(flat_sizes[i])
            strides.append(flat_strides[i])
        return Layout(tuple(sizes), tuple(strides), axes=tuple(axes))

    def _drop_layout_axes(self, layout: Layout, dropped: set[Symbol]) -> Layout:
        """Flattened sub-layout of layout with dropped axes removed."""

        assert layout.axes is not None, (
            f"_drop_layout_axes requires axes (from tile()), got {layout!r}"
        )
        flat_sizes = layout.flat_sizes
        flat_strides = (
            layout.strides if isinstance(layout.strides, tuple) else (layout.strides,)
        )
        keep = [i for i, ax in enumerate(layout.axes) if ax not in dropped]
        return Layout(
            tuple(flat_sizes[i] for i in keep),
            tuple(flat_strides[i] for i in keep),
            axes=tuple(layout.axes[i] for i in keep),
        )

    def slice(self, tensor: Tensor, bindings: Bindings) -> Tensor:
        """Fold bindings into tensor.offset and drop the bound axes from
        the (flattened) result Tensor's layout."""

        L = tensor.layout
        assert L is not None and L.axes is not None, (
            f"slice requires a layout-backed tensor, got {tensor!r}"
        )
        named = {ax for ax in L.axes if ax is not None}
        unbound = set(bindings.keys()) - named
        assert not unbound, (
            f"slice bindings {unbound!r} not present in tensor axes "
            f"{sorted(named, key=repr)!r}"
        )
        bound_axes = tuple(bindings.keys())
        new_off = tensor.offset
        if bound_axes:
            sub_layout = self._filter_layout_by_axes(L, bound_axes)
            bound_values = tuple(
                self.constant_index(v) if isinstance(v, int) else v
                for v in bindings.values()
            )
            rel = self.layout_apply(bound_values, sub_layout)
            new_off = rel if new_off is None else self.layout_sum(new_off, rel)
        new_layout = self._drop_layout_axes(L, set(bound_axes))
        return Tensor(tensor.ptr, new_off, new_layout)

    def alloc_lds_tensor(
        self, size_bytes: int, *, layout: Layout
    ) -> tuple[ir.Value, Tensor]:
        """Allocate LDS and wrap as a layout-backed Tensor."""
        handle, ptr = self.alloc_lds(size_bytes)
        return handle, Tensor(ptr, None, layout)

    def _emit_transfer_tile(
        self, tc: TiledCopy, tensor: Tensor, data: Any = None
    ) -> _TileTransfer:
        """Emit one tiled hardware transfer at tensor.ptr + tensor.offset."""
        copy_atom = tc.copy
        swizzle = tc.swizzle
        offsets = self.thread_value_offsets(tc.pid, tc.thread_layout, tc.value_layout)
        n = len(offsets)

        def _total(voff: ir.Value) -> ir.Value:
            return (
                self.layout_sum(tensor.offset, voff)
                if tensor.offset is not None
                else voff
            )

        op = getattr(self, copy_atom.op_name)
        src, dst = copy_atom.src_space, copy_atom.dst_space
        payloads: list[Any | None] = []
        tokens: list[Any] = []
        if src is MemSpace.GLOBAL and dst is MemSpace.REG:
            for voff in offsets:
                rd, tok = op(
                    tensor.ptr, dynamic_offset=self.index_to_vgpr(_total(voff))
                )
                payloads.append(rd)
                tokens.append(tok)
        elif src is MemSpace.REG and dst is MemSpace.LDS:
            assert data is not None, "reg->lds transfer needs data"
            if not isinstance(data, (list, tuple)):
                data = [data]
            assert len(data) == n, (
                f"reg->lds transfer: data has {len(data)} payloads but "
                f"value_layout.size()={n}"
            )
            for v, voff in enumerate(offsets):
                tok = op(data[v], self._addr(tensor.ptr, _total(voff), swizzle))
                payloads.append(None)
                tokens.append(tok)
        elif src is MemSpace.LDS and dst is MemSpace.REG:
            for voff in offsets:
                rd, tok = op(self._addr(tensor.ptr, _total(voff), swizzle))
                payloads.append(rd)
                tokens.append(tok)
        else:
            raise NotImplementedError(f"transfer: {src} -> {dst}")
        return _TileTransfer(payloads=tuple(payloads), tokens=tuple(tokens))

    def transfer_tiles(
        self,
        tensor: Tensor,
        tc: TiledCopy,
        *,
        unroll_axes: tuple[Symbol, ...],
        data: Optional[LayoutValues] = None,
    ) -> LayoutValues:
        """Emit one hardware transfer per (unroll_axes-coord, value-coord).

        unroll_axes: Symbols from tensor.layout to iterate over.
        data: per-tile payload bundle from a prior transfer.

        To bind some axes before iterating, pre-slice via slice(t, {...}).
        """
        L = tensor.layout
        assert L is not None and L.axes is not None, (
            f"transfer_tiles expects a Tensor layout with axes (e.g. from a tile() call), got {L!r}"
        )
        value_coords = enumerate_flat_coords(tc.value_layout.flat_sizes)
        n_per_tile = tc.value_layout.size()
        sub_layout = self._filter_layout_by_axes(L, unroll_axes)
        iter_flat = sub_layout.flat_sizes
        out_layout = Layout(
            iter_flat if n_per_tile == 1 else iter_flat + tc.value_layout.flat_sizes
        )

        def per_tile_payloads(coords):
            if data is None:
                return None
            loader_n = data.value_layout.size() // sub_layout.size()
            if loader_n == 1:
                base = data.data_at(coords)
                return list(self.split_register_range(base, n_per_tile))
            assert loader_n == n_per_tile, (
                f"data has {loader_n} payloads/tile but writer needs {n_per_tile}"
            )
            return [data.data_at(coords + vc) for vc in value_coords]

        flat_payloads: list[Any | None] = []
        flat_tokens: list[Any] = []
        for coords in enumerate_flat_coords(sub_layout.flat_sizes):
            tile_rel = self.layout_apply(
                tuple(self.constant_index(c) for c in coords), sub_layout
            )
            tile_off = (
                tile_rel
                if tensor.offset is None
                else self.layout_sum(tensor.offset, tile_rel)
            )
            sub_tensor = Tensor(tensor.ptr, tile_off)
            transfer = self._emit_transfer_tile(
                tc, sub_tensor, data=per_tile_payloads(coords)
            )
            flat_payloads.extend(transfer.payloads)
            flat_tokens.extend(transfer.tokens)
        return LayoutValues.from_flat(
            out_layout, payloads=tuple(flat_payloads), tokens=tuple(flat_tokens)
        )

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

    def transfer_tile(
        self,
        tensor: Tensor,
        tc: TiledCopy,
        *,
        data: Any = None,
    ) -> LayoutValues:
        """Emit hardware transfers for one tiled copy at ``tensor``'s offset."""
        transfer = self._emit_transfer_tile(tc, tensor, data=data)
        return LayoutValues.from_flat(
            tc.value_layout,
            payloads=transfer.payloads,
            tokens=transfer.tokens,
        )

    def copy(
        self,
        tc: TiledCopy,
        tensor: Tensor,
        data: Any = None,
    ) -> LayoutValues:
        """Deprecated: use ``transfer_tile(tensor, tc, *, data=)``."""
        return self.transfer_tile(tensor, tc, data=data)

    copy_all = transfer_tiles

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
