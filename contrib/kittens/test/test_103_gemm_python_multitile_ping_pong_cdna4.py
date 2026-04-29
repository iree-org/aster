import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tempfile

import numpy as np
import pytest

from aster import ir
from aster.dialects.kernel_builder import MFMA_F16_CDNA4
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.utils import system_has_gpu
from aster.pass_pipelines import make_default_pass_pipeline

from kittens.gemm_config import (
    A as OP_A,
    B as OP_B,
    C as OP_C,
    DIM_M,
    DIM_N,
    DIM_K,
    GemmSpec,
    GemmMappingSpec,
    OperandPath,
)
from test_102_gemm_python_multitile_cdna4 import (
    Cdna4GemmInstance,
    _build_cdna4_gemm,
)

KERNEL_NAME = "gemm_ping_pong_cdna4"


class PingPongCdna4GemmInstance(Cdna4GemmInstance):
    """Cdna4GemmInstance with ping-pong kernel name."""

    @property
    def kernel_name(self) -> str:
        return KERNEL_NAME


def compile_ping_pong_cdna4_gemm(cfg, output_hsaco_path, **kw):
    """Compile a CDNA4 ping-pong GEMM config to HSACO."""
    from aster.compiler.core import PrintOptions

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_cdna4_gemm(cfg, ping_pong_staggered=True)
        pipeline = make_default_pass_pipeline(
            num_vgprs=kw.get("num_vgprs", 256),
            num_agprs=kw.get("num_agprs", 256),
            unroll_factor_multiplier=getattr(cfg.mapping, "unroll_factor_multiplier", 1),
            epilogue_peeling=getattr(cfg.mapping, "epilogue_peeling", True),
            ll_sched=getattr(cfg.mapping, "ll_sched", False),
            hoist_iter_arg_waits=getattr(cfg.mapping, "hoist_wait", False),
        )
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=pipeline,
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=kw.get("print_ir_after_all", False),
                print_asm=kw.get("print_asm", False),
            ),
        )
    path = assemble_to_hsaco(asm, target=cfg.mapping.mcpu, wavefront_size=64, output_path=output_hsaco_path)
    assert path is not None, "assemble_to_hsaco returned None"
    return path, asm


def execute_ping_pong_cdna4_hsaco(cfg, hsaco_path, num_iterations, A, B, skip_gpu_check=False):
    """Execute a pre-compiled CDNA4 ping-pong HSACO."""
    from kittens_helpers import shuffle_weight

    mcpu = getattr(cfg.mapping, "mcpu", "gfx950")
    if not skip_gpu_check and not system_has_gpu(mcpu):
        pytest.skip(f"GPU {mcpu} not available, skip execution")

    B_gpu = shuffle_weight(B) if cfg.mapping.direct_b else B
    C_output = np.zeros(math.prod(cfg.spec.operand_shape(OP_C)), dtype=np.float32)
    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=KERNEL_NAME,
        arguments=[InputArray(A.flatten()), InputArray(B_gpu.flatten()), OutputArray(C_output)],
        grid_dim=(cfg.mapping.num_workgroups, 1, 1),
        block_dim=(cfg.mapping.num_threads, 1, 1),
        num_iterations=num_iterations,
    )
    return C_output, times_ns


def _make_instance(num_workgroups, num_waves_per_wg, num_tiles_per_wg, k_mult, pipeline_strategy=1, b_path="lds"):
    """Build a PingPongCdna4GemmInstance from tile grid and K multiplier."""
    mfma = MFMA_F16_CDNA4.shape
    twg_m, twg_n = num_tiles_per_wg[DIM_M], num_tiles_per_wg[DIM_N]
    wpw_m, wpw_n = num_waves_per_wg[DIM_M], num_waves_per_wg[DIM_N]
    assert twg_m % wpw_m == 0 and twg_n % wpw_n == 0
    M = num_workgroups[DIM_M] * twg_m * mfma[DIM_M]
    N = num_workgroups[DIM_N] * twg_n * mfma[DIM_N]
    K = k_mult * num_tiles_per_wg[DIM_K] * mfma[DIM_K]
    spec = GemmSpec.from_sizes(M, N, K, mfma_shape=list(mfma))
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=list(num_workgroups),
        num_waves_per_workgroup=list(num_waves_per_wg),
        num_tiles_per_wave=[twg_m // wpw_m, twg_n // wpw_n, num_tiles_per_wg[DIM_K]],
        pipeline_strategy=pipeline_strategy,
        operand_path=OperandPath(b_path),
        dealloc_at_read=True,
        mcpu="gfx950",
    )
    return PingPongCdna4GemmInstance(spec, mapping)


def _run_ping_pong_cdna4(cfg):
    """Compile + run a CDNA4 ping-pong GEMM, verify against numpy."""
    gs = cfg.gemm_size

    np.random.seed(42 + gs[DIM_M] + gs[DIM_N] + gs[DIM_K])
    A_mat = (np.random.randn(*cfg.spec.operand_shape(OP_A)) * 0.1).astype(np.float16)
    B_mat = (np.random.randn(*cfg.spec.operand_shape(OP_B)) * 0.1).astype(np.float16)

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
        compile_ping_pong_cdna4_gemm(cfg, tmp.name)
        C_output, _ = execute_ping_pong_cdna4_hsaco(cfg, tmp.name, 1, A_mat, B_mat)

    expected = (A_mat.astype(np.float32) @ B_mat.astype(np.float32).T).flatten()
    np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


def _min_k_iters(k_t, ps):
    from kittens_helpers import PIPELINE_STRATEGIES

    return max(PIPELINE_STRATEGIES[ps].values()) + 1


class TestPingPongCdna4Geometry:
    """Tile geometry sweep for ping-pong schedule, fixed pipeline.

    Validates 8-wave tile shapes with staggered barriers under CDNA4
    (vx4 MFMA, 2 fragments per tile, k_t=2 typical).
    """

    @pytest.mark.parametrize(
        "num_waves_per_wg,num_tiles_per_wg",
        [
            ([4, 2, 1], [8, 4, 2]),
            ([4, 2, 1], [8, 8, 2]),
            ([4, 2, 1], [16, 4, 2]),
            ([2, 4, 1], [4, 8, 2]),
            ([2, 4, 1], [8, 8, 2]),
            ([2, 4, 1], [4, 16, 2]),
        ],
        ids=[
            "8w_4x2_8x4",
            "8w_4x2_8x8",
            "8w_4x2_16x4",
            "8w_2x4_4x8",
            "8w_2x4_8x8",
            "8w_2x4_4x16",
        ],
    )
    def test_correctness(self, num_waves_per_wg, num_tiles_per_wg):
        _run_ping_pong_cdna4(
            _make_instance([1, 1, 1], num_waves_per_wg, num_tiles_per_wg, k_mult=4, pipeline_strategy=3)
        )


class TestPingPongCdna4Pipeline:
    """Pipeline strategy x k_mult sweep for ping-pong, fixed geometry.

    Tests pipeline depth interaction with K iterations under staggered
    barriers. Uses two representative 8-wave geometries.
    """

    @pytest.mark.parametrize(
        "num_waves_per_wg,num_tiles_per_wg",
        [
            ([4, 2, 1], [8, 8, 2]),
            ([2, 4, 1], [8, 8, 2]),
        ],
        ids=["8w_4x2_8x8", "8w_2x4_8x8"],
    )
    @pytest.mark.parametrize("k_mult", [4, 8], ids=["km4", "km8"])
    @pytest.mark.parametrize("pipeline_strategy", [1, 3, 5], ids=["ps1", "ps3", "ps5"])
    def test_correctness(self, num_waves_per_wg, num_tiles_per_wg, k_mult, pipeline_strategy):
        k_t = num_tiles_per_wg[DIM_K]
        if k_mult < _min_k_iters(k_t, pipeline_strategy):
            pytest.skip(f"k_mult={k_mult} < min_k_iters for ps{pipeline_strategy}")
        _run_ping_pong_cdna4(_make_instance([1, 1, 1], num_waves_per_wg, num_tiles_per_wg, k_mult, pipeline_strategy))


class TestPingPongCdna4MultiWG:
    """Multi-workgroup ping-pong correctness, orthogonal to geometry/pipeline."""

    @pytest.mark.parametrize(
        "num_workgroups,num_waves_per_wg,num_tiles_per_wg",
        [
            ([2, 2, 1], [4, 2, 1], [8, 4, 2]),
            ([2, 2, 1], [2, 4, 1], [8, 8, 2]),
            ([3, 2, 1], [4, 2, 1], [8, 8, 2]),
        ],
        ids=["mwg2x2_8w_4x2_8x4", "mwg2x2_8w_2x4_8x8", "mwg3x2_8w_4x2_8x8"],
    )
    def test_correctness(self, num_workgroups, num_waves_per_wg, num_tiles_per_wg):
        _run_ping_pong_cdna4(
            _make_instance(num_workgroups, num_waves_per_wg, num_tiles_per_wg, k_mult=4, pipeline_strategy=3)
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    parser.add_argument("--wg", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--wpw", type=int, nargs=3, default=[4, 2, 1])
    parser.add_argument("--twg", type=int, nargs=3, default=[8, 8, 2])
    parser.add_argument("--k-mult", type=int, default=2)
    parser.add_argument("--pipeline-strategy", type=int, default=1)
    parser.add_argument("--b-path", type=str, default="lds", choices=["lds", "direct_b"])
    args = parser.parse_args()

    cfg = _make_instance(args.wg, args.wpw, args.twg, args.k_mult, args.pipeline_strategy, b_path=args.b_path)
    gs = cfg.gemm_size
    tag = f"wg{'x'.join(map(str, args.wg))}_w{'x'.join(map(str, args.wpw))}_t{'x'.join(map(str, args.twg))}"
    print(f"Config: {tag}_km{args.k_mult}_ps{args.pipeline_strategy}_b{args.b_path}")
    print(f"  M={gs[DIM_M]}, N={gs[DIM_N]}, K={gs[DIM_K]}, waves={cfg.mapping.num_waves}")

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as f:
        _, asm = compile_ping_pong_cdna4_gemm(
            cfg,
            f.name,
            print_ir_after_all=args.print_ir_after_all,
            print_asm=args.print_asm,
        )
    if args.print_asm:
        print(asm)
