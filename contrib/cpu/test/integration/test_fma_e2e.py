# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import os
import shutil

import numpy as np
import pytest

from aster_cpu.execution.helpers import (
    compile_to_object,
    has_avx,
    has_avx2,
    has_avx512f,
    has_fma,
    link_shared_lib,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MLIR = os.path.join(_THIS_DIR, "fma.mlir")
_MATTR = "+avx,+avx2,+avx512f,+fma"


@pytest.fixture(scope="module")
def fma_lib(tmp_path_factory):
    """Compile fma.mlir once for all 3 tests (gated by binary existence)."""
    if not shutil.which("aster-cpu-translate"):
        pytest.skip("aster-cpu-translate not on PATH")

    tmp = tmp_path_factory.mktemp("fma")
    obj = compile_to_object(_MLIR, tmp, mattr=_MATTR)
    print((tmp / "kernel.s").read_text(), end="")
    so = link_shared_lib(obj, tmp)
    if so is None:
        pytest.skip("cannot link .so on this platform")
    return ctypes.CDLL(str(so))


def _call_fma(fma_lib, func_name: str, n_floats: int):
    a = np.arange(1, n_floats + 1, dtype=np.float32)
    b = np.arange(n_floats + 1, 2 * n_floats + 1, dtype=np.float32)
    c = np.zeros(n_floats, dtype=np.float32)

    func = getattr(fma_lib, func_name)
    func.argtypes = [ctypes.c_void_p] * 3
    func.restype = None
    func(a.ctypes.data, b.ctypes.data, c.ctypes.data)

    np.testing.assert_allclose(c, a * b, rtol=1e-6)


def test_fma_avx(fma_lib):
    if not (has_avx() and has_fma()):
        pytest.skip("host CPU lacks avx+fma")
    _call_fma(fma_lib, "test_avx_fma", 4)


def test_fma_avx2(fma_lib):
    if not (has_avx2() and has_fma()):
        pytest.skip("host CPU lacks avx2+fma")
    _call_fma(fma_lib, "test_avx2_fma", 8)


def test_fma_avx512(fma_lib):
    if not (has_avx512f() and has_fma()):
        pytest.skip("host CPU lacks avx512f+fma")
    _call_fma(fma_lib, "test_avx512_fma", 16)
