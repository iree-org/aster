# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from aster.layout import CoordTensor, Symbol


def test_coord_tensor_identity_projections():
    row, col = Symbol("row"), Symbol("col")
    proj = CoordTensor.tiled((5, 8), (1, 4), axes=(row, col)).projections
    # tile() lays out (inter..., intra...): (row, coltile, row-intra, col-intra).
    assert proj[row].flat_sizes == (5, 2, 1, 4)
    assert proj[row].strides == (1, 0, 1, 0)  # row projection: unit on row only
    assert proj[col].strides == (0, 4, 0, 1)  # col projection: coltile * 4 + intra * 1


test_coord_tensor_identity_projections()
