# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for aster.layout.int_tuple.

from aster.layout import product, suffix_product, delinearize, linearize


def test_product():
    assert product(5) == 5
    assert product((3, 4)) == 12


def test_suffix_product():
    assert suffix_product(8) == 1
    assert suffix_product((3, 2, 4)) == (8, 4, 1)


def test_delinearize():
    # First-mode-slowest (C row-major): 7 in (4,3) -> (2, 1) i.e. 7 = 2*3 + 1
    assert delinearize(7, (4, 3)) == (2, 1)
    assert delinearize(0, (4, 3)) == (0, 0)
    assert delinearize(11, (4, 3)) == (3, 2)


def test_linearize():
    # (2, 1) with strides (3, 1) -> 2*3 + 1 = 7
    assert linearize((2, 1), (3, 1)) == 7
    assert linearize((1, 2), (3, 1)) == 5
    assert linearize((3,), (2,)) == 6


def test_delinearize_linearize_roundtrip():
    sizes = (4, 3)
    strides = suffix_product(sizes)  # (3, 1) -- row-major
    for i in range(12):
        coords = delinearize(i, sizes)
        assert linearize(coords, strides) == i
