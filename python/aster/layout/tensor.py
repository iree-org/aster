# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, TypeAlias, Union, overload

if TYPE_CHECKING:
    from aster.layout.algebra import Layout, Symbol
    from aster.layout.int_tuple import IntTuple


@dataclass(frozen=True)
class CoordProj:
    """Coordinate projections with optional axis labels.

    Supports positional access and named access by axis symbol.
    """

    layouts: tuple["Layout", ...]
    axes: tuple[Optional["Symbol"], ...]

    def __post_init__(self) -> None:
        assert len(self.layouts) == len(self.axes), (
            f"projection count {len(self.layouts)} != axes count {len(self.axes)}"
        )

    def __iter__(self):
        return iter(self.layouts)

    def __len__(self) -> int:
        return len(self.layouts)

    @overload
    def __getitem__(self, key: int) -> "Layout": ...

    @overload
    def __getitem__(self, key: slice) -> tuple["Layout", ...]: ...

    @overload
    def __getitem__(self, key: "Symbol") -> "Layout": ...

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.layouts[key]
        return self.by_symbol(key)

    def by_symbol(self, axis: "Symbol") -> "Layout":
        """Return the projection layout for one axis symbol."""
        try:
            dim = self.axes.index(axis)
        except ValueError as exc:
            raise KeyError(f"unknown coordinate projection axis {axis!r}") from exc
        return self.layouts[dim]


@dataclass(frozen=True)
class CoordTensor:
    """Identity coordinate tensor used for per-dimension bounds checks.

    Each projection recovers one logical coordinate from tiled iteration
    coordinates. The origin tracks bound tile coordinates from slicing.
    """

    # Absolute coordinate of the current tile origin per logical dimension.
    # Updated by slice. Per-lane offsets are added at transfer time.
    origin: tuple[Any, ...]

    # One projection layout per logical dimension.
    projections: CoordProj

    @classmethod
    def tiled(
        cls,
        sizes: "IntTuple",
        tile_sizes: tuple,
        *,
        axes: tuple[Optional["Symbol"], ...] | None = None,
    ) -> "CoordTensor":
        """Build identity projections tiled like the data iteration layout.

        Example:
            coords = CoordTensor.tiled((m, n_pad), (1, 256), axes=(row, col))
        """
        from aster.layout.algebra import Layout, tile
        from aster.layout.int_tuple import flatten_nested

        flat = flatten_nested(sizes) if isinstance(sizes, tuple) else (sizes,)
        rank = len(flat)
        layouts = tuple(
            tile(
                Layout(flat, tuple(1 if i == d else 0 for i in range(rank))),
                tile_sizes,
                axes=axes,
            )
            for d in range(rank)
        )
        projection_axes: tuple[Optional["Symbol"], ...] = (None,) * rank
        if axes is not None:
            flat_axes = flatten_nested(axes)
            assert len(flat_axes) <= rank, (
                f"axes rank {len(flat_axes)} exceeds tensor rank {rank}"
            )
            projection_axes = flat_axes + (None,) * (rank - len(flat_axes))
        return cls(origin=(), projections=CoordProj(layouts, projection_axes))

    def projection(self, axis: "Symbol") -> "Layout":
        """Return one projection by axis symbol."""
        return self.projections[axis]

    def slice(
        self, builder: Any, bound_axes: tuple, bound_values: tuple
    ) -> CoordTensor:
        """Slice coordinate projections with the same bindings as data slicing.

        Updates origin with the bound coordinates and drops bound axes
        from each projection.
        """
        origin = self.origin or (None,) * len(self.projections)
        new_origin = []
        for d, p in enumerate(self.projections):
            rel = builder.layout_apply(
                bound_values, builder._filter_layout_by_axes(p, bound_axes)
            )
            prev = origin[d]
            new_origin.append(rel if prev is None else builder.layout_sum(prev, rel))
        return CoordTensor(
            origin=tuple(new_origin),
            projections=CoordProj(
                tuple(
                    builder._drop_layout_axes(p, set(bound_axes))
                    for p in self.projections
                ),
                self.projections.axes,
            ),
        )


@dataclass(frozen=True)
class Tensor:
    """Byte-addressed tensor.

    layout and offset are byte offsets added directly to ptr.
    Use LogicalTensor when iteration is in element indices.

    Example:
        t = Tensor(ptr, layout=Layout((M, K), (K * 4, 4)))
    """

    ptr: Any
    offset: Any = None
    layout: Optional["Layout"] = None

    def slice(self, builder: Any, bindings: "dict[Symbol, Any]") -> "Tensor":
        """Fold bindings into offset and drop the bound axes from the layout.

        ``builder`` supplies the IR-emitting helpers (layout_apply / layout_sum /
        constant_index / _filter_layout_by_axes / _drop_layout_axes).
        """
        L = self.layout
        assert L is not None and L.axes is not None, (
            f"slice requires a layout-backed tensor, got {self!r}"
        )
        named = {ax for ax in L.axes if ax is not None}
        unbound = set(bindings.keys()) - named
        assert not unbound, (
            f"slice bindings {unbound!r} not present in tensor axes "
            f"{sorted(named, key=repr)!r}"
        )
        bound_axes = tuple(bindings.keys())
        new_off = self.offset
        if bound_axes:
            bound_values = tuple(
                builder.constant_index(v) if isinstance(v, int) else v
                for v in bindings.values()
            )
            sub_layout = builder._filter_layout_by_axes(L, bound_axes)
            rel = builder.layout_apply(bound_values, sub_layout)
            new_off = rel if new_off is None else builder.layout_sum(new_off, rel)
        new_layout = builder._drop_layout_axes(L, set(bound_axes))
        return Tensor(self.ptr, new_off, new_layout)

    def byte_offset(self, builder: Any, voff: Any) -> Any:
        """Byte offset for a per-lane ``voff``: ``offset + voff`` (already bytes)."""
        if self.offset is None:
            return voff
        return builder.layout_sum(self.offset, voff)

    def buffer_voffset(self, builder: Any, buffer: Any, lane_voff: Any) -> Any:
        """Buffer voffset in bytes (a byte-addressed tensor has no per-dim OOB)."""
        return self.byte_offset(builder, lane_voff)


@dataclass(frozen=True)
class LogicalTensor:
    """Element-addressed logical tensor view.

    tensor holds ptr and iteration layout.
    elem_layout maps element indices to byte offsets and logical extents.
    coord optionally carries per-dimension coordinates for bounds checks.

    Example:
        iter_layout = tile(Layout((64, 512)), (1, 256), axes=(row, col))
        elem_layout = Layout((64, 300), (300 * 4, 4))
        t = LogicalTensor(
            Tensor(ptr, layout=iter_layout),
            elem_layout=elem_layout,
            coord=CoordTensor.tiled((64, 512), (1, 256), axes=(row, col)),
        )
    """

    tensor: Tensor
    elem_layout: Optional["Layout"] = None
    coord: Optional["CoordTensor"] = None

    def __post_init__(self) -> None:
        assert self.elem_layout is not None, (
            "LogicalTensor requires elem_layout (the element -> byte map); "
            "use a plain Tensor for a byte-addressed view"
        )

    # Expose the byte-view members directly for call sites that expect Tensor.
    @property
    def ptr(self) -> Any:
        return self.tensor.ptr

    @property
    def offset(self) -> Any:
        return self.tensor.offset

    @property
    def layout(self) -> Optional["Layout"]:
        return self.tensor.layout

    def slice(self, builder: Any, bindings: "dict[Symbol, Any]") -> "LogicalTensor":
        """Slice the inner tensor and the coordinate tensor with the same
        bindings, so the coordinate tensor stays partitioned like the data."""
        inner = self.tensor.slice(builder, bindings)
        new_coord = self.coord
        if self.coord is not None and bindings:
            bound_axes = tuple(bindings.keys())
            bound_values = tuple(
                builder.constant_index(v) if isinstance(v, int) else v
                for v in bindings.values()
            )
            new_coord = self.coord.slice(builder, bound_axes, bound_values)
        return LogicalTensor(inner, elem_layout=self.elem_layout, coord=new_coord)

    def byte_offset(self, builder: Any, voff: Any) -> Any:
        """Map the inner element-index offset to a byte offset via elem_layout."""
        return builder.layout_apply(
            self.tensor.byte_offset(builder, voff), self.elem_layout
        )

    def buffer_voffset(self, builder: Any, buffer: Any, lane_voff: Any) -> Any:
        """Buffer voffset in bytes.

        With ``coord``, predicate each inner logical
        dimension against its extent (``elem_less``); the outermost dimension is
        the contiguous tail clipped by ``buffer.num_records_bytes``. Without
        ``coord``, fall back to the plain byte offset.
        """
        if self.coord is None:
            return self.byte_offset(builder, lane_voff)
        assert buffer.num_records_bytes is not None, (
            "per-dimension bounds need a num_records_bytes sentinel"
        )
        from aster.layout.int_tuple import flatten_nested

        proj = self.coord.projections
        origin = self.coord.origin or (None,) * len(proj)
        coords = tuple(
            builder.layout_apply(lane_voff, p)
            if origin[d] is None
            else builder.layout_sum(origin[d], builder.layout_apply(lane_voff, p))
            for d, p in enumerate(proj)
        )
        byte_off = builder.layout_apply(coords, self.elem_layout)
        extents = tuple(flatten_nested(self.elem_layout.sizes))
        oob_off = builder.constant_index(buffer.num_records_bytes)
        assert len(extents) == len(coords) >= 1, "elem_layout rank mismatch"
        # Predicate each inner dimension: keep byte_off only while every inner
        # coord is in bounds, else fall to oob_off (the outermost dim is the
        # contiguous tail, already clipped by num_records).
        off = byte_off
        for coord, extent in zip(coords[1:], extents[1:]):
            ok = builder.arith_cmpi("ult", coord, builder.constant_index(extent))
            off = builder.select(ok, off, oob_off)
        return off


# A tensor argument that may be either byte-addressed or element-addressed.
TensorLike: TypeAlias = Union[Tensor, LogicalTensor]
