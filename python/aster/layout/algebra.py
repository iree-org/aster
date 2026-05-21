# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Layout class -- maps logical coordinates to physical offsets.
#
# A Layout with sizes=(sz0, sz1, ...) strides=(st0, st1, ...) computes:
#   phi(x0, x1, ...) = x0 * st0 + x1 * st1 + ...
#
# This is essentially the memref strided layout as a python class and limited
# to static / constexpr sizes and strides.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypeAlias, Union

from aster.layout.int_tuple import (
    IntTuple,
    flatten_nested,
    product,
    suffix_product,
    delinearize,
    linearize,
)

if TYPE_CHECKING:
    from aster import ir
    from aster.dialects.kernel_builder import KernelBuilder


@dataclass(frozen=True, slots=True)
class Symbol:
    """Hashable label for one bindable axis (CuTe-style named modes).

    Module-level singletons: m = Symbol("m"). Use the same object in
    tile(..., axes=...) and in local_tile(t, {m: value}) bindings.
    """

    label: str

    def __repr__(self) -> str:
        return self.label


# An axis name: either a Symbol (bindable via local_tile) or None (data axis,
# iterated by transfer_tiles but never bound). Flat, one per layout `flat_sizes`.
AxisName: TypeAlias = Union[Symbol, None]


class Layout:
    """A layout: a function from coordinates to offsets.

    Layout(sizes=(4, 16), strides=(16, 64)) creates a layout with explicit strides.
    Layout(sizes=(4, 16)) creates a compact column-major layout.

    axes (optional, set by tile()) is a flat tuple of Symbol | None
    parallel to flat_sizes. After tile(), sizes/strides take the
    two-group shape (inter, intra) and axes likewise (inter, intra).
    """

    __slots__ = ("sizes", "strides", "axes")

    sizes: IntTuple
    strides: IntTuple
    axes: Optional[tuple]

    def __init__(
        self,
        sizes: IntTuple,
        strides: IntTuple | None = None,
        *,
        axes: Optional[tuple] = None,
    ) -> None:
        self.sizes = sizes
        # TODO: support expressions with SSA values and heavy canonicalization
        self.strides = suffix_product(sizes) if strides is None else strides
        self.axes = axes

    def __call__(self, idx: int) -> int:
        """Evaluate: map integral coordinate to offset.

        Delinearize idx by sizes, then dot-product with strides.
        Handles nested tuples by flattening first.
        """
        if isinstance(self.sizes, int):
            return idx * self.strides
        flat_s = flatten_nested(self.sizes)
        flat_d = flatten_nested(self.strides)
        coords = delinearize(idx, flat_s)
        return linearize(coords, flat_d)

    def lower(
        self, b: "KernelBuilder", coord, swizzle: Optional["Swizzle"] = None
    ) -> "ir.Value":
        """Emit layout.apply (+ optional layout.swizzle) for this layout."""
        from aster.dialects import layout as layout_d

        sizes = self.sizes if isinstance(self.sizes, tuple) else (self.sizes,)
        strides = self.strides if isinstance(self.strides, tuple) else (self.strides,)
        attr = layout_d.strided_layout(list(sizes), list(strides), ctx=b._ctx)
        off = layout_d.apply(coord, attr, loc=b._loc, ip=b._kip)
        if swizzle is None:
            return off
        return layout_d.swizzle(
            off,
            bits=swizzle.bits,
            base=swizzle.base,
            shift=swizzle.shift,
            loc=b._loc,
            ip=b._kip,
        )

    def size(self) -> int:
        """Total number of logical elements."""
        return product(self.sizes)

    @property
    def flat_sizes(self) -> tuple[int, ...]:
        """All mode sizes flattened to a rank-1 tuple."""
        return flatten_nested(self.sizes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return (
            self.sizes == other.sizes
            and self.strides == other.strides
            and self.axes == other.axes
        )

    @property
    def is_flat(self) -> bool:
        """True if all modes are scalar (non-nested)."""
        if isinstance(self.sizes, int):
            return True
        return all(isinstance(s, int) for s in self.sizes)

    def __len__(self) -> int:
        if isinstance(self.sizes, tuple):
            return len(self.sizes)
        return 1

    def __str__(self) -> str:
        return f"{self.sizes}:{self.strides}"

    def __repr__(self) -> str:
        return f"Layout(sizes={self.sizes},strides={self.strides})"


class Swizzle:
    """XOR-based swizzle: offset = idx ^ ((idx >> shift) & mask).

    Used for LDS bank conflict avoidance. Applied after layout evaluation.
    """

    __slots__ = ("bits", "base", "shift")

    def __init__(self, bits: int, base: int, shift: int) -> None:
        self.bits = bits
        self.base = base
        self.shift = shift

    def __call__(self, offset: int) -> int:
        mask = ((1 << self.bits) - 1) << self.base
        return offset ^ ((offset >> self.shift) & mask)

    def __repr__(self) -> str:
        return f"Swizzle(bits={self.bits}, base={self.base}, shift={self.shift})"


class SwizzledLayout:
    """Layout with an additive unswizzled layout plus an XOR-swizzled layout.

    Evaluating layout_apply(coord, SwizzledLayout(...)) is equivalent to:
        linearize(coord, unswizzled_base)
            + swizzle_spec(linearize(coord, swizzled_base))

    Used for G2S two-part LDS addressing where a row-stride correction is
    added unswizzled and the tile-local offset is XOR-swizzled.

    All constructor arguments are keyword-only and required:
        unswizzled_base: layout for the additive (unswizzled) term.
        swizzled_base:   layout for the term that is XOR-swizzled.
        swizzle_spec:    Swizzle parameters (bits/base/shift).
    """

    __slots__ = ("unswizzled_base", "swizzled_base", "swizzle_spec")

    def __init__(
        self,
        *,
        unswizzled_base: Layout,
        swizzled_base: Layout,
        swizzle_spec: Swizzle,
    ) -> None:
        self.unswizzled_base = unswizzled_base
        self.swizzled_base = swizzled_base
        self.swizzle_spec = swizzle_spec

    def __repr__(self) -> str:
        return (
            f"SwizzledLayout(unswizzled_base={self.unswizzled_base}, "
            f"swizzled_base={self.swizzled_base}, "
            f"swizzle_spec={self.swizzle_spec})"
        )

    def lower(
        self, b: "KernelBuilder", coord, swizzle: Optional["Swizzle"] = None
    ) -> "ir.Value":
        """Emit the two-part address: unswizzled base + XOR-swizzled base."""
        from aster import ir

        assert swizzle is None, "SwizzledLayout already carries its own swizzle"
        unswizzled_off = self.unswizzled_base.lower(b, coord)
        swizzled_off = self.swizzled_base.lower(b, coord, self.swizzle_spec)
        d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
        return b.affine_apply(d0 + d1, [unswizzled_off, swizzled_off])


def make_layout(*layouts: Layout) -> Layout:
    """Combine layouts into one: each becomes a mode of the result."""
    sizes, strides = zip(*((a.sizes, a.strides) for a in layouts))
    return Layout(sizes=sizes, strides=strides)


def enumerate_flat_coords(
    flat_sizes: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return an enumeration of all coordinates under flat_sizes's row-major flat order."""
    return tuple(delinearize(idx, flat_sizes) for idx in range(product(flat_sizes)))


def flat_index(coord: tuple[int, ...], layout: Layout) -> int:
    """Linear index for coord under layout's row-major flat order."""
    flat_s = layout.flat_sizes
    assert len(coord) == len(flat_s), (
        f"coord rank {len(coord)} != layout flat rank {len(flat_s)}"
    )
    return linearize(coord, suffix_product(flat_s))


def tile(
    layout: Layout,
    tile_sizes: tuple,
    *,
    axes: tuple[Optional[Symbol], ...] | None = None,
) -> Layout:
    """Tile layout into a flat Layout.

    When axes are provided, attach Symbols to the layout's axes and
    complete with None axes for the remaining sizes.
    """
    flat_sizes = (
        flatten_nested(layout.sizes)
        if isinstance(layout.sizes, tuple)
        else (layout.sizes,)
    )
    flat_strides = (
        flatten_nested(layout.strides)
        if isinstance(layout.strides, tuple)
        else (layout.strides,)
    )
    assert len(tile_sizes) == len(flat_sizes), (
        f"tile_sizes rank {len(tile_sizes)} != layout rank {len(flat_sizes)}"
    )

    inter_sizes: list[int] = []
    inter_strides: list[int] = []
    intra_sizes: list[int] = []
    intra_strides: list[int] = []
    for i, (size, stride, ts) in enumerate(zip(flat_sizes, flat_strides, tile_sizes)):
        if isinstance(ts, int):
            assert size % ts == 0, f"tile dim {ts} does not divide layout dim {size}"
            intra_sizes.append(ts)
            intra_strides.append(stride)
            inter_sizes.append(size // ts)
            inter_strides.append(ts * stride)
        else:
            assert isinstance(ts, tuple), (
                f"tile_sizes element must be int or tuple, got {ts!r}"
            )
            intra_sizes.append(ts[0])
            intra_strides.append(stride)
            cumulative = ts[0]
            for tj in ts[1:]:
                inter_sizes.append(tj)
                inter_strides.append(cumulative * stride)
                cumulative *= tj
            assert size % cumulative == 0, (
                f"hierarchical tile {ts} (cumulative {cumulative}) does not "
                f"divide layout dim {size}"
            )
            inter_sizes.append(size // cumulative)
            inter_strides.append(cumulative * stride)

    all_flattened_sizes = tuple(inter_sizes) + tuple(intra_sizes)
    all_flattened_strides = tuple(inter_strides) + tuple(intra_strides)
    # Flatten axes (unpacking any hierarchical sub-tuples), then complete
    # with None for the intra-tile axes.
    all_flattened_axes = None
    if axes is not None:
        flat_axes = flatten_nested(axes)
        all_flattened_axes = flat_axes + (None,) * (
            len(all_flattened_sizes) - len(flat_axes)
        )
    return Layout(all_flattened_sizes, all_flattened_strides, axes=all_flattened_axes)
