# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for aster.layout.int_tuple.

from aster.layout import product, prefix_product, delinearize, linearize


def test_product():
    assert product(5) == 5
    assert product((3, 4)) == 12


def test_prefix_product():
    assert prefix_product(8) == 1
    assert prefix_product((3, 2, 4)) == (1, 3, 6)


def test_delinearize():
    # 7 in col-major (4,3): 7 = 3 + 4*1
    assert delinearize(7, (4, 3)) == (3, 1)
    assert delinearize(0, (4, 3)) == (0, 0)
    assert delinearize(11, (4, 3)) == (3, 2)


def test_linearize():
    # (3, 1) with strides (1, 4) -> 3 + 4 = 7
    assert linearize((3, 1), (1, 4)) == 7
    assert linearize((2, 1), (1, 4)) == 6
    assert linearize((3,), (2,)) == 6


def test_delinearize_linearize_roundtrip():
    sizes = (4, 3)
    strides = prefix_product(sizes)
    for i in range(12):
        coords = delinearize(i, sizes)
        assert linearize(coords, strides) == i
