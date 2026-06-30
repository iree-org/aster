"""Pipelined 128x128x128 f16 subtile GEMM for gfx1250 (4 waves, TDM + WMMA)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest

from aster import ir
from aster.dialects.kernel_builder_with_layouts import (
    KernelBuilderWithLayouts as KernelBuilder,
)
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

from aster.layout import Layout

MCPU = "gfx1250"
KERNEL_NAME = "gemm_subtile"
WAVE_SIZE = 32

STAGE_LOAD = 0
STAGE_WRITE = 1
STAGE_READ = 2
STAGE_COMPUTE = 3
N_STAGES = STAGE_COMPUTE + 1

TILE_K = 32
WMMA_MNEMONIC = "v_wmma_f32_16x16x32_f16"

# Fixed: 1 WG, 4 waves (2x2), 4x4 WMMA subtiles per wave -> 128x128 C.
WPW_M, WPW_N = 2, 2
SUB_M, SUB_N = 4, 4
K_DIM = 128

M_TOTAL = WPW_M * SUB_M * 16
N_TOTAL = WPW_N * SUB_N * 16
STRIP_M = SUB_M * 16
STRIP_N = SUB_N * 16
FRAG_BYTES = 16 * TILE_K * 2
NUM_THREADS = WPW_M * WPW_N * WAVE_SIZE


def _build_gemm_subtile_gfx1250(K, target=MCPU):
    assert K % TILE_K == 0, f"K={K} must be divisible by {TILE_K}"
    k_tiles = K // TILE_K
    assert k_tiles >= N_STAGES, f"k_tiles={k_tiles} < N_STAGES={N_STAGES}"

    stride_a = K * 2
    stride_b = K * 2
    stride_c = N_TOTAL * 4
    num_threads = WPW_M * WPW_N * WAVE_SIZE

    lds_total_a = M_TOTAL * TILE_K * 2
    lds_total_b = N_TOTAL * TILE_K * 2

    GLOBAL_STORE_TILE_C = Layout((2, 16, 8), (8 * stride_c, 4, stride_c))
    GLOBAL_STORE_SUB_TILE_C = Layout((SUB_M, SUB_N), (16 * stride_c, 16 * 4))

    b = KernelBuilder("gemm_subtile_mod", KERNEL_NAME, target=target, wave_size=WAVE_SIZE)
    b.set_grid_dims(1)
    b.set_block_dims(num_threads)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    a_ptr, b_ptr, c_ptr = b.load_args()

    wid = b.wave_id()
    wave_m_idx, wave_n_idx = b.delinearize_index(wid, (WPW_M, WPW_N))

    d0 = ir.AffineExpr.get_dim(0)
    d1 = ir.AffineExpr.get_dim(1)
    a_wave_row = b.affine_apply(d0 * STRIP_M, [wave_m_idx])
    b_wave_row = b.affine_apply(d0 * STRIP_N, [wave_n_idx])

    n_accs = SUB_M * SUB_N
    acc_inits = [b.init_vgprx8(b.constant_i32(0)) for _ in range(n_accs)]
    accs_final = []

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)

    @b.loop(c0, b.constant_index(k_tiles), c1, iter_args=acc_inits, results=accs_final)
    def body(k_iv, *accs):
        accs = list(accs)
        k_byte_off = b.affine_apply(d0 * (TILE_K * 2), [k_iv])

        with b.stage(STAGE_LOAD):
            lds_a_h, lds_a = b.alloc_lds(lds_total_a)
            lds_b_h, lds_b = b.alloc_lds(lds_total_b)
            a_lds_off = b.affine_apply(d0 * (STRIP_M * TILE_K * 2), [wave_m_idx])
            a_glb_off = b.affine_apply(d0 + d1 * stride_a, [k_byte_off, a_wave_row])
            a_g0, a_g1, a_g2, a_g3 = b.make_tdm_descriptor(
                lds_offset=b.affine_apply(d0 + d1, [lds_a, a_lds_off]),
                global_addr=a_ptr,
                global_byte_offset=a_glb_off,
                element_bytes=2,
                dim0=K,
                dim1=STRIP_M,
                tile0=TILE_K,
                tile1=STRIP_M,
                stride=K,
            )
            a_tdm_tok = b.tensor_load_to_lds(a_g0, a_g1, a_g2, a_g3)
            b_lds_off = b.affine_apply(d0 * (STRIP_N * TILE_K * 2), [wave_n_idx])
            b_glb_off = b.affine_apply(d0 + d1 * stride_b, [k_byte_off, b_wave_row])
            b_g0, b_g1, b_g2, b_g3 = b.make_tdm_descriptor(
                lds_offset=b.affine_apply(d0 + d1, [lds_b, b_lds_off]),
                global_addr=b_ptr,
                global_byte_offset=b_glb_off,
                element_bytes=2,
                dim0=K,
                dim1=STRIP_N,
                tile0=TILE_K,
                tile1=STRIP_N,
                stride=K,
            )
            b_tdm_tok = b.tensor_load_to_lds(b_g0, b_g1, b_g2, b_g3)

        with b.stage(STAGE_WRITE):
            b.wait_deps_gfx1250(a_tdm_tok, b_tdm_tok)
            b.s_barrier_signal()
            b.s_barrier_wait()

        with b.stage(STAGE_READ):
            WMMA_FRAG_LAYOUT = Layout((2, 16), (16 * 2, TILE_K * 2))
            laneoff = b.layout_apply(b.lane_id(), WMMA_FRAG_LAYOUT)
            f0, f1, f2 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1), ir.AffineExpr.get_dim(2)
            a_regs = []
            for mi in range(SUB_M):
                frag_base = b.affine_apply(
                    f0 + f1 + f2 + (mi * FRAG_BYTES), [lds_a, a_lds_off, laneoff]
                )
                a_lo, a_lo_tok = b.ds_load_b128(frag_base)
                a_hi, a_hi_tok = b.ds_load_b128(frag_base, b.constant_i32(16))
                b.wait_deps_gfx1250(a_lo_tok, a_hi_tok)
                a_regs.append(
                    b._make_register_range(
                        list(b.split_register_range(a_lo, 4)) + list(b.split_register_range(a_hi, 4))
                    )
                )

        with b.stage(STAGE_COMPUTE):
            for ni in range(SUB_N):
                frag_base = b.affine_apply(
                    f0 + f1 + f2 + (ni * FRAG_BYTES), [lds_b, b_lds_off, laneoff]
                )
                b_lo, b_lo_tok = b.ds_load_b128(frag_base)
                b_hi, b_hi_tok = b.ds_load_b128(frag_base, b.constant_i32(16))
                b.wait_deps_gfx1250(b_lo_tok, b_hi_tok)
                b_frag = b._make_register_range(
                    list(b.split_register_range(b_lo, 4)) + list(b.split_register_range(b_hi, 4))
                )
                for mi in range(SUB_M):
                    accs[mi * SUB_N + ni] = b.mfma(WMMA_MNEMONIC, accs[mi * SUB_N + ni], a_regs[mi], b_frag)
            b.dealloc_lds(lds_a_h)
            b.dealloc_lds(lds_b_h)
        return accs

    c_tile_off = b.affine_apply(d0 * (STRIP_M * stride_c) + d1 * (STRIP_N * 4), [wave_m_idx, wave_n_idx])
    flat_regs = []
    for acc in accs_final:
        flat_regs += list(b.split_register_range(acc, 8))
    b.store_multi_fragment_to_global(
        b._make_register_range(flat_regs),
        c_ptr,
        c_tile_off,
        GLOBAL_STORE_TILE_C,
        GLOBAL_STORE_SUB_TILE_C,
        b.global_store_b32,
        nt=False,
    )
    return b.build()


def _run_gemm_subtile_gfx1250(print_opts=None, target=MCPU):
    """Compile, assemble, and (if a gfx1250 GPU is present) execute + check.

    Returns (status, asm): status is "ok", "no-assembler", or "no-gpu".
    """
    np.random.seed(42 + M_TOTAL + N_TOTAL + K_DIM)
    A = (np.random.randn(M_TOTAL, K_DIM) * 0.1).astype(np.float16)
    B = (np.random.randn(N_TOTAL, K_DIM) * 0.1).astype(np.float16)
    C_output = np.zeros(M_TOTAL * N_TOTAL, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_gemm_subtile_gfx1250(K_DIM, target)
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
            print_opts=print_opts,
        )

    path = assemble_to_hsaco(asm, target=target, wavefront_size=WAVE_SIZE)
    if path is None:
        return "no-assembler", asm

    with hsaco_file(path):
        if not system_has_mcpu(mcpu=target):
            return "no-gpu", asm
        execute_hsaco(
            hsaco_path=path,
            kernel_name=KERNEL_NAME,
            arguments=[
                InputArray(A.flatten().view(np.uint16)),
                InputArray(B.flatten().view(np.uint16)),
                OutputArray(C_output),
            ],
            grid_dim=(1, 1, 1),
            block_dim=(NUM_THREADS, 1, 1),
        )

    expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
    np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
    return "ok", asm


class TestSubtileGfx1250:
    def test_correctness(self):
        status, _ = _run_gemm_subtile_gfx1250()
        if status == "no-assembler":
            pytest.skip(f"{MCPU} HSACO did not assemble")
        if status == "no-gpu":
            pytest.skip(f"{MCPU} GPU not available")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    args = parser.parse_args()

    from aster.compiler.core import PrintOptions

    status, asm = _run_gemm_subtile_gfx1250(
        print_opts=PrintOptions.from_flags(
            print_ir_after_all=args.print_ir_after_all,
            print_asm=args.print_asm,
        ),
    )
    if args.print_asm:
        print(asm)
    if status == "no-assembler":
        print(f"FAIL: {MCPU} HSACO did not assemble")
        sys.exit(1)
    elif status == "no-gpu":
        print(f"{MCPU} GPU not available; hsaco assembled OK")
    else:
        print(f"PASS: subtile GEMM {M_TOTAL}x{N_TOTAL}x{K_DIM} matches reference")
