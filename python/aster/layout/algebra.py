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
# Only flat (non-nested) sizes/strides for now.
# This is essentially the memref strided layout as a python class and limited
# to static / constexpr sizes and strides.
#
# TODO: Evolve to more advanced forms

from __future__ import annotations

from aster.layout.int_tuple import IntTuple, product, prefix_product


class Layout:
    """A layout: a function from coordinates to offsets.

    Layout(sizes=(4, 16), strides=(16, 64)) creates a layout with explicit strides.
    Layout(sizes=(4, 16)) creates a compact column-major layout.
    """

    __slots__ = ("sizes", "strides")

    sizes: IntTuple
    strides: IntTuple

    def __init__(self, sizes: IntTuple, strides: IntTuple | None = None) -> None:
        self.sizes = sizes
        # TODO: support expressions with SSA values and heavy canonicalization
        self.strides = prefix_product(sizes) if strides is None else strides

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return self.sizes == other.sizes and self.strides == other.strides

    def __len__(self) -> int:
        if isinstance(self.sizes, tuple):
            return len(self.sizes)
        return 1

    def __str__(self) -> str:
        return f"{self.sizes}:{self.strides}"

    def __repr__(self) -> str:
        return f"Layout(sizes={self.sizes},strides={self.strides})"


def make_layout(*layouts: Layout) -> Layout:
    """Combine layouts into one: each becomes a mode of the result."""
    sizes, strides = zip(*((a.sizes, a.strides) for a in layouts))
    return Layout(sizes=sizes, strides=strides)
