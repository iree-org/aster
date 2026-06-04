# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, overload

if TYPE_CHECKING:
    from aster.layout.algebra import Layout, Symbol
    from aster.layout.int_tuple import IntTuple


@dataclass(frozen=True)
class CoordProj:
    """Collection of coordinate projections with symbol-based access."""

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
        """Return the coordinate projection layout for a named logical axis."""
        try:
            dim = self.axes.index(axis)
        except ValueError as exc:
            raise KeyError(f"unknown coordinate projection axis {axis!r}") from exc
        return self.layouts[dim]


@dataclass(frozen=True)
class CoordTensor:
    """Companion to a tensor, that tracks the per-dimension coordinate for OOB."""

    # origin: the absolute coordinate of the bound tile origin, accumulated per
    # dim as slice() binds the same axes the data is sliced on (empty until
    # sliced). The per-lane part is added at transfer time.
    origin: tuple[Any, ...]

    # one coordinate-projection Layout per logical dim (identity tensor basis),
    # with both positional and symbol-based lookup.
    projections: CoordProj

    @classmethod
    def tiled(
        cls,
        sizes: "IntTuple",
        tile_sizes: tuple,
        *,
        axes: tuple[Optional["Symbol"], ...] | None = None,
    ) -> "CoordTensor":
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
        """Return the coordinate projection layout for a named logical axis."""
        return self.projections[axis]

    def slice(
        self, builder: Any, bound_axes: tuple, bound_values: tuple
    ) -> CoordTensor:
        """Partition identically to the data: bind the same axes in each
        projection (folding the tile origin into ``origin`` per dim) and drop
        them.  ``builder`` supplies layout_apply / layout_sum for IR emission."""
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
    """View Tensor: pointer + dynamic offset + optional Layout."""

    ptr: Any
    offset: Any = None
    layout: Optional["Layout"] = None
