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
    "07_simple_fusion",
)
sys.path.insert(0, _EX)
from simple_fusion import run_simple_fusion  # noqa: E402


@pytest.mark.parametrize("size", [4])
def test_simple_fusion(size):
    run_simple_fusion(size=size)
