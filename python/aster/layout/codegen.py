# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Layout to MLIR op plan: Delinearize + Linearize class representation.

# TODO: Generalize to support SSA values of type index

from __future__ import annotations

from dataclasses import dataclass

from .algebra import Layout


@dataclass(frozen=True, slots=True)
class Delinearize:
    """Decompose a 1-D index into multi-dim coordinates.

    Corresponds to affine.delinearize_index %idx into (basis...).
    """

    basis: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Linearize:
    """Combine multi-dim coordinates into a 1-D offset with explicit strides.

    Lowered via affine.apply with map: (d0, d1, ...) -> (d0*s0 + d1*s1 + ...).
    Note: affine.linearize_index cannot be used here because it interprets its
    basis as sizes (computing strides as suffix products), not as explicit strides.
    """

    basis: tuple[int, ...]


def layout_to_ops(layout: Layout) -> list[Delinearize | Linearize]:
    """Convert a Layout to a [Delinearize, Linearize] op plan.

    The plan describes how to map a 1-D thread index to a 1-D byte offset:
      1. Delinearize the index by the layout's shape
      2. Linearize the resulting coordinates by the layout's strides
    """
    if not isinstance(layout, Layout):
        raise TypeError(f"Expected Layout, got {type(layout)}")

    sizes = layout.sizes if isinstance(layout.sizes, tuple) else (layout.sizes,)
    strides = layout.strides if isinstance(layout.strides, tuple) else (layout.strides,)
    assert isinstance(sizes, tuple) and isinstance(strides, tuple)

    return [Delinearize(basis=sizes), Linearize(basis=strides)]
