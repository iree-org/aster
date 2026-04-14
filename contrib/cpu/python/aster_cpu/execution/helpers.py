# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""End-to-end execution helpers for contrib/cpu AMX kernels."""

import ctypes
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

_RUNTIME_C = Path(__file__).parent / "amx_runtime.c"


def has_amx_bf16() -> bool:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return False
    try:
        return "amx_bf16" in Path("/proc/cpuinfo").read_text()
    except OSError:
        return False


def _compile_to_object(mlir_filename: os.PathLike, tmp_path: Path) -> Path:
    asm = tmp_path / "amx.s"
    obj = tmp_path / "amx.o"

    aster_cpu_translate = shutil.which("aster-cpu-translate")
    assert aster_cpu_translate, "aster-cpu-translate not on PATH"
    llvm_install = os.environ.get("LLVM_INSTALL")
    assert llvm_install, "LLVM_INSTALL not set (activate the worktree venv)"
    llvm_mc = Path(llvm_install) / "bin" / "llvm-mc"

    with asm.open("wb") as f:
        subprocess.run(
            [aster_cpu_translate, "--mlir-to-amx-asm", str(mlir_filename)],
            stdout=f,
            check=True,
        )
    subprocess.run(
        [
            str(llvm_mc),
            "-triple=x86_64-unknown-linux-gnu",
            "-mattr=+amx-tile,+amx-bf16",
            "-filetype=obj",
            str(asm),
            "-o",
            str(obj),
        ],
        check=True,
    )
    return obj


def _link_shared_lib(obj_path: Path, tmp_path: Path) -> Optional[Path]:
    if platform.system() != "Linux":
        return None
    so = tmp_path / "libamx.so"
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    assert cc, "no system C compiler on PATH"
    subprocess.run(
        [cc, "-shared", "-fPIC", "-O2", str(_RUNTIME_C), str(obj_path), "-o", str(so)],
        check=True,
    )
    return so


def compile_and_run(
    mlir_filename: os.PathLike,
    function_name: str,
    *,
    input_data: List[np.ndarray],
    output_data: List[np.ndarray],
    verify_fn: Callable[[List[np.ndarray], List[np.ndarray]], None],
    tmp_path: Path,
    print_asm: bool = False,
) -> None:
    import pytest

    obj_path = _compile_to_object(mlir_filename, tmp_path)

    if print_asm:
        asm_path = tmp_path / "amx.s"
        print(asm_path.read_text(), end="")

    so_path = _link_shared_lib(obj_path, tmp_path)
    if so_path is None:
        pytest.skip(
            "cross-compilation to x86_64 ELF succeeded but host cannot "
            f"natively link an ELF .so ({platform.system()})"
        )
    if not has_amx_bf16():
        pytest.skip(
            "cross-compilation succeeded but host CPU lacks amx_bf16 "
            "(requires Sapphire Rapids or newer; see /proc/cpuinfo)"
        )

    lib = ctypes.CDLL(str(so_path))
    func = getattr(lib, function_name)
    func.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64]
    func.restype = None

    a, b = input_data
    (c,) = output_data
    func(a.ctypes.data, b.ctypes.data, c.ctypes.data, 64)

    verify_fn(input_data, output_data)
