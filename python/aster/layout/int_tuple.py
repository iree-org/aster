# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Integer tuple utilities for the layout algebra.
#
# A shape/stride is either an int (rank-1) or a flat tuple of ints.
# No nesting support -- will be added when needed.

from __future__ import annotations

from functools import reduce
from typing import TypeAlias

# A shape or stride: scalar int or flat tuple of ints.
# TODO: support ir.Value of type index everywhere
IntTuple: TypeAlias = int | tuple[int, ...]


def product(a: IntTuple) -> int:
    """Product of all elements.

    Works on int or flat tuple. Pure-Python equivalent of
    affine.linearize_index with all-ones coordinates (i.e. computes the
    total domain size).
    """
    if isinstance(a, tuple):
        return reduce(lambda x, y: x * y, a, 1)
    return a


def suffix_product(a: IntTuple) -> IntTuple:
    """Exclusive suffix product: (s0, s1, s2) -> (s1*s2, s2, 1).

    Gives compact row-major (C-order) strides from sizes.
    Matches upstream MLIR's first-mode-slowest convention.
    """
    if isinstance(a, tuple):
        r: list[int] = []
        p = 1
        for v in reversed(a):
            r.append(p)
            p *= v
        r.reverse()
        return tuple(r)
    return 1


def delinearize(idx: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Decompose a 1-D index into multi-dim coordinates by sizes.

    Uses first-mode-slowest convention (C row-major), matching upstream
    MLIR's affine.delinearize_index. The first size is the outermost
    dimension. E.g. delinearize(7, (4, 3)) -> (2, 1)  i.e. 7 = 2*3 + 1
    """
    coords: list[int] = []
    for s in reversed(sizes):
        coords.append(idx % s)
        idx //= s
    coords.reverse()
    return tuple(coords)


def linearize(coords: tuple[int, ...], strides: tuple[int, ...]) -> int:
    """Combine multi-dim coordinates into a 1-D offset via strides.

    Pure-Python equivalent of affine.linearize_index.     linearize((3,
    1), (1, 4)) -> 7
    """
    assert len(coords) == len(strides)
    return sum(c * d for c, d in zip(coords, strides))
