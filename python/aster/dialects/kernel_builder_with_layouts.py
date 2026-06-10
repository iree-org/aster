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
from aster.layout.tensor import Tensor, TensorLike
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
buffer_load_dwordx4 = Copy("buffer_load_dwordx4", MemSpace.GLOBAL, MemSpace.REG)
buffer_store_dwordx4 = Copy("buffer_store_dwordx4", MemSpace.REG, MemSpace.GLOBAL)
ds_write_64b = Copy("ds_write_b64", MemSpace.REG, MemSpace.LDS)
ds_read_64b = Copy("ds_read_b64", MemSpace.LDS, MemSpace.REG)
g2s_buffer_load_dwordx4 = Copy("g2s_buffer_load_dwordx4", MemSpace.GLOBAL, MemSpace.LDS)
global_store_dword = Copy("global_store_dword", MemSpace.REG, MemSpace.GLOBAL)

# gfx1250 equivalents (different mnemonics, same logical operations).
global_load_b128 = Copy("global_load_b128", MemSpace.GLOBAL, MemSpace.REG)
ds_store_64b = Copy("ds_store_b64", MemSpace.REG, MemSpace.LDS)
global_store_b32 = Copy("global_store_b32", MemSpace.REG, MemSpace.GLOBAL)


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
class TransferTileG2S:
    """G2S resources for transfer_tiles_g2s, built by prepare_transfer_tiles_g2s.

    per_thread_voff rationale:
    G2S hw writes M0 + TID*16 linearly to LDS, to account for this constraint:
        - align the global source address to match the LDS address with a
        swizzled tile-local read, and
        - adjust the global source address by an unswizzled row-stride correction.
    Note: Atm m0 is a named register and not an SSA value.
    """

    rsrc: Any
    soff: Any
    m0: Any
    per_thread_voff: Any


@dataclass(frozen=True, slots=True)
class TransferTileBuffer:
    """Buffer transfer resources.

    num_records_bytes is the byte bound from ptr used by the SRD. Use
    with LogicalTensor.coord for per-dimension bounds checks.
    """

    rsrc: Any
    soff: Any
    num_records_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class _TileTransfer:
    """Per-value payloads and tokens emitted for one tiled transfer."""

    payloads: tuple[Any | None, ...]
    tokens: tuple[Any, ...]


class KernelBuilderWithLayouts(KernelBuilder):
    """KernelBuilder with layout-first tile operations."""

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

    def slice(self, tensor: TensorLike, bindings: Bindings) -> TensorLike:
        """Fold bindings into the tensor's offset and drop the bound axes.

        Polymorphic: Tensor.slice / LogicalTensor.slice each return their own
        type (a LogicalTensor also slices its coordinate tensor).
        """
        return tensor.slice(self, bindings)

    def alloc_lds_tensor(
        self, size_bytes: int, *, layout: Layout
    ) -> tuple[ir.Value, Tensor]:
        """Allocate LDS and wrap as a layout-backed Tensor."""
        handle, ptr = self.alloc_lds(size_bytes)
        return handle, Tensor(ptr, None, layout)

    def _emit_transfer_tile(
        self,
        tc: TiledCopy,
        tensor: TensorLike,
        data: Any = None,
        buffer: Optional[TransferTileBuffer] = None,
        *,
        nt: bool = True,
        fence_token: Optional[ir.Value] = None,
    ) -> _TileTransfer:
        """Emit one tiled hardware transfer at tensor.ptr + tensor.offset.

        fence_token: is an optional fence_token to pin the memory operations onto.
        """
        copy_atom = tc.copy
        swizzle = tc.swizzle
        offsets = self.thread_value_offsets(tc.pid, tc.thread_layout, tc.value_layout)
        n = len(offsets)
        op = getattr(self, copy_atom.op_name)
        src, dst = copy_atom.src_space, copy_atom.dst_space
        payloads: list[Any | None] = []
        tokens: list[Any] = []
        if src is MemSpace.GLOBAL and dst is MemSpace.REG:
            assert fence_token is None, (
                "global->reg transfer does not support fence_token"
            )
            if buffer is not None:
                # SRD bound handles ragged tail.
                for voff in offsets:
                    off = self.index_to_vgpr(tensor.buffer_voffset(self, buffer, voff))
                    load = self._buffer_load_op(
                        copy_atom.op_name,
                        self.alloc_vgprx4(),
                        buffer.rsrc,
                        buffer.soff,
                        off,
                        nt=nt,
                    )
                    payloads.append(load.results[0])
                    tokens.append(load.results[1])
            else:
                for voff in offsets:
                    off = self.index_to_vgpr(tensor.byte_offset(self, voff))
                    rd, tok = op(tensor.ptr, dynamic_offset=off)
                    payloads.append(rd)
                    tokens.append(tok)
        elif src is MemSpace.REG and dst is MemSpace.GLOBAL:
            assert fence_token is None, (
                "reg->global transfer does not support fence_token"
            )
            assert data is not None, "reg->global transfer needs data"
            if not isinstance(data, (list, tuple)):
                data = [data]
            assert len(data) == n, (
                f"reg->global transfer: data has {len(data)} payloads but "
                f"value_layout.size()={n}"
            )
            if buffer is not None:
                # SRD bound handles ragged tail.
                for v, voff in enumerate(offsets):
                    off = self.index_to_vgpr(tensor.buffer_voffset(self, buffer, voff))
                    tok = op(data[v], buffer.rsrc, buffer.soff, off, nt=nt)
                    payloads.append(None)
                    tokens.append(tok)
            else:
                # Keep direct global store behavior used by existing kernels.
                for v, voff in enumerate(offsets):
                    off = self.index_to_vgpr(tensor.byte_offset(self, voff))
                    tok = op(data[v], tensor.ptr, dynamic_offset=off, nt=True)
                    payloads.append(None)
                    tokens.append(tok)
        elif src is MemSpace.REG and dst is MemSpace.LDS:
            assert fence_token is None, "reg->lds transfer does not support fence_token"
            assert data is not None, "reg->lds transfer needs data"
            if not isinstance(data, (list, tuple)):
                data = [data]
            assert len(data) == n, (
                f"reg->lds transfer: data has {len(data)} payloads but "
                f"value_layout.size()={n}"
            )
            for v, voff in enumerate(offsets):
                addr = self._addr(tensor.ptr, tensor.byte_offset(self, voff), swizzle)
                tok = op(data[v], addr)
                payloads.append(None)
                tokens.append(tok)
        elif src is MemSpace.LDS and dst is MemSpace.REG:
            for voff in offsets:
                addr = self._addr(tensor.ptr, tensor.byte_offset(self, voff), swizzle)
                rd, tok = op(addr, fence_token=fence_token)
                payloads.append(rd)
                tokens.append(tok)
        else:
            raise NotImplementedError(f"transfer: {src} -> {dst}")
        return _TileTransfer(payloads=tuple(payloads), tokens=tuple(tokens))

    def transfer_tiles(
        self,
        tensor: TensorLike,
        tc: TiledCopy,
        *,
        unroll_axes: tuple[Symbol, ...],
        data: Optional[LayoutValues] = None,
        buffer: Optional[TransferTileBuffer] = None,
        nt: bool = True,
        fence_token: Optional[ir.Value] = None,
    ) -> LayoutValues:
        """Emit one transfer per unrolled tile coordinate and value coordinate.

        Use slice to bind some axes before calling this API.

        fence_token: is an optional fence_token to pin the memory operations onto.
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
                tc,
                sub_tensor,
                data=per_tile_payloads(coords),
                buffer=buffer,
                nt=nt,
                fence_token=fence_token,
            )
            flat_payloads.extend(transfer.payloads)
            flat_tokens.extend(transfer.tokens)
        return LayoutValues.from_flat(
            out_layout, payloads=tuple(flat_payloads), tokens=tuple(flat_tokens)
        )

    def prepare_transfer_tiles_buffer(
        self,
        ptr: ir.Value,
        *,
        buffer_num_records_bytes: int,
        srd_flags: int = 0x24924,
    ) -> TransferTileBuffer:
        """Build loop-invariant SRD resources for buffer load and store.

        buffer_num_records_bytes sets the byte bound from ptr.

        Example:
            buf = b.prepare_transfer_tiles_buffer(ptr, buffer_num_records_bytes=nbytes)
        """
        nr = self.s_mov_b32(buffer_num_records_bytes)
        rsrc = self.make_buffer_rsrc(ptr, nr, self.constant_i32(0), flags=srd_flags)
        return TransferTileBuffer(
            rsrc=rsrc,
            soff=self.s_mov_b32(0),
            num_records_bytes=buffer_num_records_bytes,
        )

    def prepare_transfer_tiles_g2s(
        self,
        ptr: ir.Value,
        *,
        buffer_num_records_bytes: int,
        spatial_dim: int,
        tile_row_bytes: int,
        global_load_bytes: int,
        global_row_stride: int,
        swizzle: Swizzle,
        srd_flags: int = 0x24924,
    ) -> TransferTileG2S:
        """Set up the loop-invariant G2S resources for transfer_tiles_g2s.

        Bundles the four things a G2S load needs into a TransferTileG2S:
          - buffer resource descriptor over `ptr` (num records in bytes),
          - a zero scalar offset (SGPR),
          - the M0 register (LDS destination base),
          - a per-lane source byte offset within one tile.

        M0 is allocated here, outside any `stage` scope, so the pipeliner does
        not treat it as stage-rotating state. (Allocating it inside
        transfer_tiles_g2s, which runs under a `stage` context, stamps
        sched.stage on the alloca and desyncs the s_mov from the G2S load.)
        """
        nr = self.s_mov_b32(buffer_num_records_bytes)
        rsrc = self.make_buffer_rsrc(ptr, nr, self.constant_i32(0), flags=srd_flags)
        soff = self.s_mov_b32(0)
        m0 = self.alloc_m0()

        lanes_per_row = self.wave_size // spatial_dim
        tile_local = Layout(
            (spatial_dim, lanes_per_row), (tile_row_bytes, global_load_bytes)
        )
        row_corr = Layout(
            (spatial_dim, lanes_per_row), (global_row_stride - tile_row_bytes, 0)
        )
        lane = self.lane_id()
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        per_thread_voff = self.affine_apply(
            d0 + d1,
            [
                self.layout_apply(lane, tile_local, swizzle=swizzle),
                self.layout_apply(lane, row_corr),
            ],
        )
        return TransferTileG2S(
            rsrc=rsrc, soff=soff, m0=m0, per_thread_voff=per_thread_voff
        )

    def transfer_tiles_g2s(
        self,
        src_tensor: Tensor,
        dst_tensor: Tensor,
        g2s: TransferTileG2S,
        *,
        unroll_axes: tuple,
    ) -> tuple:
        """Emit G2S transfers (`buffer_load_dwordx4_lds`) per (unroll_axes) coord.

        G2S goes global -> LDS in one instruction via MUBUF semantics: the
        hardware writes `M0 + TID*16` to LDS, with the global source byte
        coming from `g2s.rsrc + g2s.soff + g2s.per_thread_voff + tile_global`.
        This does not fit the FLAT TiledCopy path so it has its own entry.

        Args:
          src_tensor: global source. `src_tensor.offset` is the wave-and-iter
            global byte base; `src_tensor.layout` gives per-tile global byte
            strides (must contain `unroll_axes`).
          dst_tensor: LDS destination, mirror of src_tensor: offset is the
            wave LDS byte base, layout gives per-tile LDS byte strides.
          g2s: prepared resources from prepare_transfer_tiles_g2s.
          unroll_axes: Symbol axes to iterate over (one G2S each).

        Returns:
          A tuple of G2S write tokens, one per (unroll_axes) coord, in
          row-major order. Wait on these before reading from LDS.
        """
        src_L = src_tensor.layout
        dst_L = dst_tensor.layout
        assert src_L is not None and src_L.axes is not None, (
            f"src_tensor needs a layout with axes, got {src_L!r}"
        )
        assert dst_L is not None and dst_L.axes is not None, (
            f"dst_tensor needs a layout with axes, got {dst_L!r}"
        )
        src_sub = self._filter_layout_by_axes(src_L, unroll_axes)
        dst_sub = self._filter_layout_by_axes(dst_L, unroll_axes)
        s0, s1, s2 = (ir.AffineExpr.get_symbol(i) for i in range(3))
        tokens: list[ir.Value] = []
        for coords in enumerate_flat_coords(src_sub.flat_sizes):
            coord_vals = tuple(self.constant_index(c) for c in coords)
            tile_g_off = self.layout_apply(coord_vals, src_sub)
            tile_l_off = self.layout_apply(coord_vals, dst_sub)
            voff = self.affine_apply(
                s0 + s1 + s2,
                [],
                [src_tensor.offset, tile_g_off, g2s.per_thread_voff],
            )
            lds_rel = (
                self.layout_sum(dst_tensor.offset, tile_l_off)
                if dst_tensor.offset is not None
                else tile_l_off
            )
            lds_off = self.layout_sum(dst_tensor.ptr, lds_rel)
            self.set_m0(g2s.m0, self.index_to_sgpr(lds_off))
            tok = self.g2s_buffer_load_dwordx4(
                g2s.m0, g2s.rsrc, g2s.soff, self.index_to_vgpr(voff)
            )
            tokens.append(tok)
        return tuple(tokens)

    def _scope_count(self, scope: Scope) -> int:
        if scope is Scope.LANE:
            return self.wave_size
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
        tensor: TensorLike,
        tc: TiledCopy,
        *,
        data: Any = None,
        buffer: Optional[TransferTileBuffer] = None,
        nt: bool = True,
        fence_token: Optional[ir.Value] = None,
    ) -> LayoutValues:
        """Emit hardware transfers for one tiled copy at ``tensor``'s offset."""
        transfer = self._emit_transfer_tile(
            tc, tensor, data=data, buffer=buffer, nt=nt, fence_token=fence_token
        )
        return LayoutValues.from_flat(
            tc.value_layout,
            payloads=transfer.payloads,
            tokens=transfer.tokens,
        )

    def copy(
        self,
        tc: TiledCopy,
        tensor: TensorLike,
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
        fence_token: Optional[ir.Value] = None,
    ) -> list[tuple[ir.Value, ir.Value]]:
        """Read MFMA fragments from LDS with swizzle.

        read_fn(addr, fence_token=fence_token) -> (data, tok).
        fence_token: is an optional fence_token to pin the memory operations onto.

        Returns a list of (data, tok) pairs, one per sub-tile.
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
            results.append(read_fn(addr, fence_token=fence_token))
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
        # Split accumulator into individual registers.
        # split_register_range with n_subs = register count gives individual regs.
        from aster._mlir_libs._amdgcn import AGPRRangeType

        if isinstance(acc.type, AGPRRangeType):
            agprs = self.split_ax4(acc)
        else:
            from aster.dialects._amdgcn_ops_gen import SplitRegisterRangeOp

            op = SplitRegisterRangeOp(input=acc, loc=self._loc, ip=self._kip)
            agprs = tuple(op.results_)
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
