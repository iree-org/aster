"""Multi-tile ping-pong GEMM: test_102 with staggered barriers.

Reuses _build_multitile_gemm(ping_pong_staggered=True) which adds:
  1. s_barrier after COMPUTE (inside pipelined K-loop)
  2. conditional s_barrier after the loop (waves < 4 only)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math

import numpy as np
import pytest

from aster import ir
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
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
)
from test_102_gemm_python_multitile import (
    MultitileGemmInstance,
    _build_multitile_gemm,
)

KERNEL_NAME = "gemm_ping_pong"


class PingPongGemmInstance(MultitileGemmInstance):
    """MultitileGemmInstance with ping-pong kernel name."""

    @property
    def kernel_name(self):
        return KERNEL_NAME


def compile_ping_pong_gemm(cfg, output_hsaco_path, **kw):
    """Compile a ping-pong GEMM config to HSACO."""
    from aster.compiler.core import PrintOptions

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_multitile_gemm(cfg, ping_pong_staggered=True)
        pipeline = make_default_pass_pipeline(
            num_vgprs=kw.get("num_vgprs", 256),
            num_agprs=kw.get("num_agprs", 256),
            unroll_factor_multiplier=getattr(cfg.mapping, "unroll_factor_multiplier", 1),
            epilogue_peeling=getattr(cfg.mapping, "epilogue_peeling", True),
            ll_sched=getattr(cfg.mapping, "ll_sched", False),
            hoist_iter_arg_waits=getattr(cfg.mapping, "hoist_wait", False),
            set_mfma_priority=getattr(cfg.mapping, "set_mfma_priority", True),
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


def execute_ping_pong_hsaco(cfg, hsaco_path, num_iterations, A, B, skip_gpu_check=False):
    """Execute a pre-compiled HSACO."""
    from aster.execution.utils import system_has_gpu

    mcpu = getattr(cfg.mapping, "mcpu", "gfx942")
    if not skip_gpu_check and not system_has_gpu(mcpu):
        pytest.skip(f"GPU {mcpu} not available, skip execution")

    C_output = np.zeros(math.prod(cfg.spec.operand_shape(OP_C)), dtype=np.float32)
    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=KERNEL_NAME,
        arguments=[InputArray(A.flatten()), InputArray(B.flatten()), OutputArray(C_output)],
        grid_dim=(cfg.mapping.num_workgroups, 1, 1),
        block_dim=(cfg.mapping.num_threads, 1, 1),
        num_iterations=num_iterations,
    )
    return C_output, times_ns


def _make_instance(num_workgroups, num_waves_per_wg, num_tiles_per_wg, k_mult, pipeline_strategy=1):
    """Build a PingPongGemmInstance from list parameters."""
    assert num_tiles_per_wg[DIM_M] % num_waves_per_wg[DIM_M] == 0
    assert num_tiles_per_wg[DIM_N] % num_waves_per_wg[DIM_N] == 0
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=list(num_workgroups),
        num_waves_per_workgroup=list(num_waves_per_wg),
        num_tiles_per_wave=[
            num_tiles_per_wg[DIM_M] // num_waves_per_wg[DIM_M],
            num_tiles_per_wg[DIM_N] // num_waves_per_wg[DIM_N],
            num_tiles_per_wg[DIM_K],
        ],
        pipeline_strategy=pipeline_strategy,
    )
    probe_spec = GemmSpec.from_sizes(1, 1, 1)
    mfma = probe_spec.mfma_shape
    tile_k_elems = (mapping.wave_size // mfma[DIM_M]) * (mapping.global_load_bytes // probe_spec.elt_bytes_a)
    M = num_workgroups[DIM_M] * num_tiles_per_wg[DIM_M] * mfma[DIM_M]
    N = num_workgroups[DIM_N] * num_tiles_per_wg[DIM_N] * mfma[DIM_N]
    k = k_mult * num_tiles_per_wg[DIM_K] * tile_k_elems
    return PingPongGemmInstance(GemmSpec.from_sizes(M, N, k), mapping)


def _run_ping_pong(cfg):
    """Compile + run, verify against numpy."""
    gs = cfg.gemm_size
    np.random.seed(42 + gs[DIM_M] + gs[DIM_N] + gs[DIM_K])
    A_mat = (np.random.randn(*cfg.spec.operand_shape(OP_A)) * 0.1).astype(np.float16)
    B_mat = (np.random.randn(*cfg.spec.operand_shape(OP_B)) * 0.1).astype(np.float16)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_multitile_gemm(cfg, ping_pong_staggered=True)
        asm = compile_mlir_module_to_asm(module, pass_pipeline=make_default_pass_pipeline())

    mcpu = cfg.mapping.mcpu
    path = assemble_to_hsaco(asm, target=mcpu, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler not compiled with {mcpu} support")

    C_output = np.zeros(math.prod(cfg.spec.operand_shape(OP_C)), dtype=np.float32)
    with hsaco_file(path):
        if not system_has_mcpu(mcpu=mcpu):
            pytest.skip(f"{mcpu} GPU not available")
        execute_hsaco(
            hsaco_path=path,
            kernel_name=KERNEL_NAME,
            arguments=[InputArray(A_mat.flatten()), InputArray(B_mat.flatten()), OutputArray(C_output)],
            grid_dim=(cfg.mapping.num_workgroups, 1, 1),
            block_dim=(cfg.mapping.num_threads, 1, 1),
        )

    expected = (A_mat.astype(np.float32) @ B_mat.astype(np.float32).T).flatten()
    np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


def _min_k_iters(twg_k, ps):
    from kittens_helpers import PIPELINE_STRATEGIES as PS

    return max(PS[ps].values()) + 1


class TestPythonGEMMPingPong:
    @pytest.mark.parametrize(
        "num_workgroups,num_waves_per_wg,num_tiles_per_wg",
        [
            # 8 waves (4x2)
            ([1, 1, 1], [4, 2, 1], [8, 4, 1]),
            ([1, 1, 1], [4, 2, 1], [8, 6, 1]),
            ([1, 1, 1], [4, 2, 1], [8, 8, 1]),
            ([1, 1, 1], [4, 2, 1], [12, 4, 1]),
            ([1, 1, 1], [4, 2, 1], [12, 6, 1]),
            ([1, 1, 1], [4, 2, 1], [12, 8, 1]),
            # 8 waves (2x4)
            ([1, 1, 1], [2, 4, 1], [4, 8, 1]),
            ([1, 1, 1], [2, 4, 1], [6, 8, 1]),
            ([1, 1, 1], [2, 4, 1], [8, 8, 1]),
            ([1, 1, 1], [2, 4, 1], [8, 12, 1]),
            ([1, 1, 1], [2, 4, 1], [12, 8, 1]),
            # Multi-WG (8 waves)
            ([2, 2, 1], [4, 2, 1], [8, 4, 1]),
            ([2, 2, 1], [2, 4, 1], [8, 8, 1]),
            ([3, 2, 1], [4, 2, 1], [8, 6, 1]),
        ],
        ids=[
            "8w_4x2_8x4",
            "8w_4x2_8x6",
            "8w_4x2_8x8",
            "8w_4x2_12x4",
            "8w_4x2_12x6",
            "8w_4x2_12x8",
            "8w_2x4_4x8",
            "8w_2x4_6x8",
            "8w_2x4_8x8",
            "8w_2x4_8x12",
            "8w_2x4_12x8",
            "mwg2x2_8w_4x2_8x4",
            "mwg2x2_8w_2x4_8x8",
            "mwg3x2_8w_4x2_8x6",
        ],
    )
    @pytest.mark.parametrize("k_mult", [2, 4, 8], ids=["km2", "km4", "km8"])
    @pytest.mark.parametrize("pipeline_strategy", [1, 3, 5], ids=["ps1", "ps3", "ps5"])
    def test_correctness(
        self,
        num_workgroups,
        num_waves_per_wg,
        num_tiles_per_wg,
        k_mult,
        pipeline_strategy,
    ):
        k_t = num_tiles_per_wg[DIM_K]
        if k_mult < _min_k_iters(k_t, pipeline_strategy):
            pytest.skip(f"k_mult={k_mult} < min_k_iters for ps{pipeline_strategy}")
        _run_ping_pong(_make_instance(num_workgroups, num_waves_per_wg, num_tiles_per_wg, k_mult, pipeline_strategy))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    parser.add_argument("--wg", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--wpw", type=int, nargs=3, default=[4, 2, 1])
    parser.add_argument("--twg", type=int, nargs=3, default=[8, 8, 1])
    parser.add_argument("--k-mult", type=int, default=4)
    parser.add_argument("--pipeline-strategy", type=int, default=1)
    args = parser.parse_args()

    cfg = _make_instance(args.wg, args.wpw, args.twg, args.k_mult, args.pipeline_strategy)
    gs = cfg.gemm_size
    tag = f"wg{'x'.join(map(str, args.wg))}_w{'x'.join(map(str, args.wpw))}_t{'x'.join(map(str, args.twg))}"
    print(f"Config: {tag}_km{args.k_mult}_ps{args.pipeline_strategy}")
    print(f"  M={gs[DIM_M]}, N={gs[DIM_N]}, K={gs[DIM_K]}")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as f:
        _, asm = compile_ping_pong_gemm(
            cfg,
            f.name,
            print_ir_after_all=args.print_ir_after_all,
            print_asm=args.print_asm,
        )
    if args.print_asm:
        print(asm)
