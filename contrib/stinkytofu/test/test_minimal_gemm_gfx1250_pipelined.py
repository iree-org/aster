# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Minimal pipelined gfx1250 GEMM with TDM and StinkyTofu handoff."""

import os
import sys
import tempfile
import shutil
import subprocess

import numpy as np
import pytest

from aster import ir
from aster.dialects.kernel_builder_with_layouts import (
    KernelBuilderWithLayouts as KernelBuilder,
    global_load_b128,
    ds_store_64b,
)
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig
from aster.pass_pipelines_stinkytofu import make_stinkytofu_handoff_pipeline

from aster.layout import Layout, Swizzle, Tensor

MCPU = "gfx1250"
WAVE_SIZE = 32  # gfx1250 WMMA is Wave32.

STAGE_LOAD = 0
STAGE_WRITE = 1
STAGE_READ = 2
STAGE_COMPUTE = 3

TILE_K = 32
TILE_N = 16


def _build_gemm_pipelined_gfx1250(M, N, K):
    # fmt: off
    assert M == 16, f"M={M} must be 16"
    assert N == 16, f"N={N} must be 16"
    assert K % 32 == 0, f"K={K} must be divisible by 32"
    k_tiles = K // 32
    stride_a = K * 2       # K * bf16 = 2 bytes per element
    stride_b_elements = K  # K elements for the tensor descriptor
    stride_c = N * 4       # N * f32 = N * 4 bytes per element
    # fmt: on
    # wmma_mnemonic over Wave32 WMMA 16x16 f32 accumulator fragment
    wmma_mnemonic = "v_wmma_f32_16x16x32_bf16"
    #     -> 8f32 per thread:
    #   lane l holds column l % 16
    #     within each lange, register r in 0..7 holds row (l // 16) * 8 + r
    #
    # Note: the wmma fragment is column-major per lane but C is row-major in global memory.
    GLOBAL_STORE_TILE_C = Layout((2, 16, 8), (8 * stride_c, 4, stride_c))
    GLOBAL_STORE_SUB_TILE_C = Layout(1, 0)

    b = KernelBuilder(
        "gemm_gfx1250_mod", "gemm_gfx1250", target=MCPU, wave_size=WAVE_SIZE
    )
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    # A operand: global_load_b128 -> ds_store_b64 (VGPR path).
    tc_load_a = b.make_tiled_copy_descriptor(
        global_load_b128,
        thread_layout=Layout((16, 2), (stride_a, 32)),
        value_layout=Layout(2, 16),  # 2 x 16 * f16 = 2 x load_128b per thread
    )
    tc_dsw_a = b.make_tiled_copy_descriptor(
        ds_store_64b,
        thread_layout=Layout((16, 2), (64, 32)),
        value_layout=Layout(4, 8),  # 4 x 8 * f16 = 4 x store_64b per thread
        swizzle=Swizzle(bits=3, base=3, shift=3),
    )

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    acc_init = b.init_vgprx8(b.constant_i32(0))

    results = []

    @b.loop(c0, b.constant_index(k_tiles), c1, iter_args=[acc_init], results=results)  # fmt: skip
    def _(k_iv, acc):
        d0 = ir.AffineExpr.get_dim(0)  # k_iv
        tile_off = b.affine_apply(d0 * 64, [k_iv])  # one K-tile = 64B for A and B

        with b.stage(STAGE_LOAD):
            lds_a_h, lds_a = b.alloc_lds(1024)
            lds_b_h, lds_b = b.alloc_lds(1024)
            # A: global -> VGPR via global_load_b128.
            A_tile = Tensor(a_ptr, offset=tile_off)
            a_load_res = b.transfer_tile(A_tile, tc_load_a)
            # B: global -> LDS via TDM, rebuilt per iteration at b_ptr + tile_off.
            tdm_g0, tdm_g1, tdm_g2, tdm_g3 = b.make_tdm_descriptor(
                lds_offset=lds_b,
                global_addr=b_ptr,
                global_byte_offset=tile_off,
                element_bytes=2,  # bf16
                dim0=K,
                dim1=TILE_N,
                tile0=TILE_K,
                tile1=TILE_N,
                stride=stride_b_elements,
            )
            b_tdm_tok = b.tensor_load_to_lds(tdm_g0, tdm_g1, tdm_g2, tdm_g3)

        with b.stage(STAGE_WRITE):
            b.wait_deps_gfx1250(a_load_res)
            sA = Tensor(lds_a)
            # Two b128 loads -> four b64 stores, in value-layout order.
            a_data = list(b.split_vx4(a_load_res.data_at(0))) + list(
                b.split_vx4(a_load_res.data_at(1))
            )
            dsw_a_res = b.transfer_tile(sA, tc_dsw_a, data=a_data)

        with b.stage(STAGE_READ):
            b.wait_deps_gfx1250(dsw_a_res, b_tdm_tok)
            a_lo, a_tok_lo = b.ds_load_b128(lds_a)
            a_hi, a_tok_hi = b.ds_load_b128(lds_a, b.constant_i32(16))
            bt_lo, b_tok_lo = b.ds_load_b128(lds_b)
            bt_hi, b_tok_hi = b.ds_load_b128(lds_b, b.constant_i32(16))

        with b.stage(STAGE_COMPUTE):
            b.wait_deps_gfx1250(a_tok_lo, a_tok_hi, b_tok_lo, b_tok_hi)
            a_lo_regs = b.split_register_range(a_lo, 4)
            a_hi_regs = b.split_register_range(a_hi, 4)
            a_frag = b._make_register_range(list(a_lo_regs) + list(a_hi_regs))
            b_lo_regs = b.split_register_range(bt_lo, 4)
            b_hi_regs = b.split_register_range(bt_hi, 4)
            b_frag = b._make_register_range(list(b_lo_regs) + list(b_hi_regs))
            acc = b.mfma(wmma_mnemonic, acc, a_frag, b_frag)
            b.dealloc_lds(lds_a_h)
            b.dealloc_lds(lds_b_h)

        return [acc]

    acc_final = results[0]
    b.store_multi_fragment_to_global(
        acc_final,
        c_ptr,
        b.constant_index(0),
        GLOBAL_STORE_TILE_C,
        GLOBAL_STORE_SUB_TILE_C,
        b.global_store_b32,
        nt=False,
    )
    return b.build()


def _handoff_asm_to_stinkytofu(asm, *, kernel_name="gemm_gfx1250", print_opts=None):
    """Run stinkytofu-opt on labeled ASTER asm (schedule + wait insertion)."""
    st_opt = shutil.which("stinkytofu-opt")
    if st_opt is None:
        return "no-stinkytofu", None

    # StinkyTofu passes run on the region between --from-label/--to-label;
    # bracket the whole kernel body (kernel label to s_endpgm).
    lines = []
    for line in asm.splitlines():
        if line.strip() == "s_endpgm":
            lines.append("aster_handoff_end:")
        lines.append(line)
        if line.strip() == f"{kernel_name}:":
            lines.append("aster_handoff_begin:")
    labeled_asm = "\n".join(lines) + "\n"

    with tempfile.TemporaryDirectory() as tmp:
        s_path = os.path.join(tmp, "handoff.s")
        out_path = os.path.join(tmp, "scheduled.s")
        with open(s_path, "w") as f:
            f.write(labeled_asm)
        # Per-pass IR dumps land in cwd as before.txt/after.txt.
        result = subprocess.run(
            [
                st_opt,
                "--arch",
                "gfx1250",
                s_path,
                "--StinkyBuildImplicitDependencyPass",
                "--StinkyDAGSchedulerPass",
                "--StinkyWaitCntInsertionPass",
                "--emit-asm",
                "--from-label",
                "aster_handoff_begin",
                "--to-label",
                "aster_handoff_end",
                "-o",
                out_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp,
        )
        if result.returncode != 0:
            raise RuntimeError(f"stinkytofu-opt failed:\n{result.stderr}")
        if print_opts is not None and getattr(print_opts, "print_ir_after_all", False):
            with open(os.path.join(tmp, "after.txt")) as f:
                print(f.read())
        with open(out_path) as f:
            st_asm = f.read()
    return "ok", st_asm


def _run_gemm_pipelined_gfx1250(
    M,
    N,
    K,
    print_opts=None,
    config=PipelineConfig(),
    *,
    stinkytofu_handoff=False,
):
    """Compile the pipelined gfx1250 GEMM kernel."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_gemm_pipelined_gfx1250(M, N, K)
        pass_pipeline = (
            make_stinkytofu_handoff_pipeline(config)
            if stinkytofu_handoff
            else make_default_pass_pipeline(config)
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=pass_pipeline,
            print_opts=print_opts,
        )

    if stinkytofu_handoff:
        return _handoff_asm_to_stinkytofu(asm, print_opts=print_opts)

    import ml_dtypes

    bf16 = ml_dtypes.bfloat16
    np.random.seed(42 + M + N + K)
    A = (np.random.randn(M, K) * 0.1).astype(bf16)
    B = (np.random.randn(N, K) * 0.1).astype(bf16)
    C_output = np.zeros(M * N, dtype=np.float32)

    path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=WAVE_SIZE)
    if path is None:
        return "no-assembler", asm

    with hsaco_file(path):
        if not system_has_mcpu(mcpu=MCPU):
            return "no-gpu", asm
        execute_hsaco(
            hsaco_path=path,
            kernel_name="gemm_gfx1250",
            arguments=[
                InputArray(A.flatten().view(np.uint16)),
                InputArray(B.flatten().view(np.uint16)),
                OutputArray(C_output),
            ],
            grid_dim=(1, 1, 1),
            block_dim=(WAVE_SIZE, 1, 1),
        )

    expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
    np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
    return "ok", asm


class TestMinimalGemmGfx1250Pipelined:
    @pytest.mark.parametrize("M, N, K", [(16, 16, 128), (16, 16, 256)])
    def test_gemm_pipelined_gfx1250(self, M, N, K):
        pytest.importorskip("ml_dtypes")
        status, _ = _run_gemm_pipelined_gfx1250(M, N, K)
        if status == "no-assembler":
            pytest.skip(f"LLVM assembler not compiled with {MCPU} support")
        if status == "no-gpu":
            pytest.skip(f"{MCPU} GPU not available")

    @pytest.mark.parametrize("k", [128])
    def test_gemm_handoff_to_stinkytofu(self, k):
        """End-to-end: ASTER wait-free ASM -> stinkytofu-opt raise, schedule, insert waits."""
        status, st_asm = _run_gemm_pipelined_gfx1250(16, 16, k, stinkytofu_handoff=True)
        if status == "no-stinkytofu":
            pytest.skip("stinkytofu-opt not found in PATH")
        assert st_asm is not None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    parser.add_argument("--stinkytofu-handoff", action="store_true")
    parser.add_argument("--ll-sched", type=int, default=1)
    args = parser.parse_args()

    from aster.compiler.core import PrintOptions

    print_opts = PrintOptions.from_flags(
        print_ir_after_all=args.print_ir_after_all,
        print_asm=args.print_asm,
    )
    if args.stinkytofu_handoff:
        status, asm = _run_gemm_pipelined_gfx1250(
            args.M, args.N, args.K, print_opts, stinkytofu_handoff=True
        )
        if status == "no-stinkytofu":
            print("FAIL: stinkytofu-opt not found in PATH")
            sys.exit(1)
        if args.print_asm:
            print(asm)
        print(f"PASS: handoff M={args.M} N={args.N} K={args.K} scheduled by StinkyTofu")
        sys.exit(0)

    status, asm = _run_gemm_pipelined_gfx1250(
        args.M,
        args.N,
        args.K,
        print_opts,
        config=PipelineConfig(ll_sched=args.ll_sched),
    )
    if args.print_asm:
        print(asm)
    if status == "no-assembler":
        print(f"FAIL: LLVM assembler not compiled with {MCPU} support")
        sys.exit(1)
    elif status == "no-gpu":
        print(f"{MCPU} GPU not available; hsaco assembled OK")
    else:
        print(f"PASS: GEMM M={args.M} N={args.N} K={args.K} matches reference")
