# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

import pytest

_EX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
    "06_copy_layout",
)
sys.path.insert(0, _EX)
from copy_2d_padded import run_copy_2d_padded  # noqa: E402


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_copy_2d_padded(depth):
    run_copy_2d_padded(m=64, n=300, depth=depth, num_cu=8)
