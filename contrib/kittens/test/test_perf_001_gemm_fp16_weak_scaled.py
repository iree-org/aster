"""Test: Correctness gate for weak-scaled constexpr GEMM (16x16x16 MFMA + dwordx4).

Verifies multi-WG, multi-wave, multi-tile GEMM at K=128 against numpy reference.
Uses v_mfma_f32_16x16x16_f16: 16x16 output tiles, K=16 per MFMA.
Global loads use dwordx4 (16x32 transfer tiles): 2x bandwidth vs dwordx2.

Tiles are specified per-workgroup (num_tiles_per_wg). Per-wave tile counts
are derived: num_tiles_per_wave = num_tiles_per_wg / num_waves_per_wg.
"""

from dataclasses import dataclass

import numpy as np
from kittens.gemm_config import (
    A as OP_A,
    B as OP_B,
    C as OP_C,
    DIM_M,
    DIM_N,
    DIM_K,
    GemmSpec,
    GemmMappingSpec,
    LoadType,
    Operand,
    OperandPath,
    WeakScaledMappedGemmInstance,
)
import pytest
import tempfile

from aster.pass_pipelines import make_default_pass_pipeline

from kittens_helpers import (
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
    constexpr_substitutions_16x32,
    shuffle_weight,
    MCPU,
    LDS_SIZE,
    WAVEFRONT_SIZE,
)

# Keyed by (b_path, load_type). b_path: "lds", "direct_b", or "direct_ab".
# load_type: "flat" or "buffer".
KERNEL_NAMES = {
    "lds": "gemm_f16_weak_scaled",
    "direct_b": "gemm_f16_direct_b",
    "direct_ab": "gemm_f16_direct_ab",
}
MLIR_FILES = {
    ("lds", "flat"): "test_perf_001_gemm_fp16_weak_scaled.mlir",
    ("lds", "buffer"): "test_perf_001_gemm_fp16_weak_scaled.mlir",
    ("direct_b", "flat"): "test_perf_001_gemm_fp16_direct_b.mlir",
    ("direct_b", "buffer"): "test_perf_001_gemm_fp16_direct_b.mlir",
    ("direct_ab", "flat"): "test_perf_001_gemm_fp16_direct_ab.mlir",
    ("direct_ab", "buffer"): "test_perf_001_gemm_fp16_direct_ab.mlir",
}
# Both flat and buffer use the same helpers: after PR #418 the _buf helpers were
# unified into the flat helpers file via !aster_utils.any type-erasure.
K_LOOP_HELPERS_FILES = {
    "flat": "gemm_16x32_f16_k_loop_helpers.mlir",
    "buffer": "gemm_16x32_f16_k_loop_helpers.mlir",
}

_A_STAGES_TO_STRATEGY = {1: 0, 2: 1, 3: 3, 4: 5, 5: 7, 6: 9}


def _make_weak_scaled_mapped_gemm_instance(
    num_workgroups_per_kernel,
    num_waves_per_wg,
    num_tiles_per_wg,
    *,
    k,
    a_stages=2,
    pipeline_strategy=-1,
    load_type="flat",
    b_path="lds",
    num_wg_per_cu=1,
    mfma_shape=None,
    **mapping_kwargs,
):
    """Helper to build a WeakScaledMappedGemmInstance from list weak-scale parameters."""
    if mfma_shape is None:
        mfma_shape = [16, 16, 16]
    if pipeline_strategy < 0:
        pipeline_strategy = _A_STAGES_TO_STRATEGY[a_stages]
    M = num_workgroups_per_kernel[DIM_M] * num_tiles_per_wg[DIM_M] * mfma_shape[DIM_M]
    N = num_workgroups_per_kernel[DIM_N] * num_tiles_per_wg[DIM_N] * mfma_shape[DIM_N]
    spec = GemmSpec.from_sizes(M, N, k, mfma_shape=mfma_shape)
    mapping = GemmMappingSpec(
        num_workgroups_per_kernel=list(num_workgroups_per_kernel),
        num_waves_per_workgroup=list(num_waves_per_wg),
        num_tiles_per_wave=[
            num_tiles_per_wg[DIM_M] // num_waves_per_wg[DIM_M],
            num_tiles_per_wg[DIM_N] // num_waves_per_wg[DIM_N],
            num_tiles_per_wg[DIM_K] // num_waves_per_wg[DIM_K],
        ],
        pipeline_strategy=pipeline_strategy,
        load_type=LoadType(load_type),
        operand_path=OperandPath(b_path),
        num_wg_per_cu=num_wg_per_cu,
        **mapping_kwargs,
    )
    return WeakScaledMappedGemmInstance(spec, mapping)


def _load_k_loop_helpers(load_type="flat", b_path="lds"):
    """Read the shared K-loop helper functions MLIR fragment."""
    helpers_path = get_mlir_file(K_LOOP_HELPERS_FILES[load_type])
    with open(helpers_path) as f:
        helpers = f.read()
    if b_path == "direct_b":
        direct_b_path = get_mlir_file("gemm_16x32_f16_k_loop_helpers_direct_b.mlir")
        with open(direct_b_path) as f:
            helpers += "\n" + f.read()
    elif b_path == "direct_ab":
        direct_ab_path = get_mlir_file("gemm_16x32_f16_k_loop_helpers_direct_ab.mlir")
        with open(direct_ab_path) as f:
            helpers += "\n" + f.read()
    return helpers


def _make_substitutions(cfg):
    """Build template substitutions dict for a WeakScaledMappedGemmInstance."""
    subs = {"{{K_LOOP_HELPERS}}": _load_k_loop_helpers(cfg.load_type, cfg.b_path)}
    subs.update(
        constexpr_substitutions_16x32(
            cfg.mapping.num_tiles_per_wave[DIM_M],
            cfg.mapping.num_tiles_per_wave[DIM_N],
            cfg.gemm_size[DIM_K],
            cfg.pipeline_strategy,
        )
    )
    subs["{{M_WG}}"] = str(cfg.mapping.num_workgroups_per_kernel[DIM_M])
    subs["{{N_WG}}"] = str(cfg.mapping.num_workgroups_per_kernel[DIM_N])
    subs["{{M_WAVES}}"] = str(cfg.mapping.num_waves_per_workgroup[DIM_M])
    subs["{{N_WAVES}}"] = str(cfg.mapping.num_waves_per_workgroup[DIM_N])
    subs["{{M_TILES_WG}}"] = str(cfg.mapping.num_tiles_per_workgroup[DIM_M])
    subs["{{N_TILES_WG}}"] = str(cfg.mapping.num_tiles_per_workgroup[DIM_N])
    subs["{{A_LDS_BYTES}}"] = str(
        cfg.mapping.num_tiles_per_workgroup[DIM_M]
        * cfg.mapping.num_tiles_per_wave[DIM_K]
        * 1024
    )
    subs["{{B_LDS_BYTES}}"] = str(
        cfg.mapping.num_tiles_per_wave[DIM_K]
        * cfg.mapping.num_tiles_per_workgroup[DIM_N]
        * 1024
    )
    gs = cfg.gemm_size
    subs["{{STRIDE_C}}"] = str(gs[DIM_N] * 4)  # f32 = 4 bytes
    subs["{{SHARED_MEM}}"] = "0"
    subs["{{NUM_THREADS}}"] = str(cfg.num_threads)
    subs["{{NUM_BLOCKS}}"] = str(cfg.num_workgroups)
    subs["{{K_T}}"] = str(cfg.mapping.num_tiles_per_wave[DIM_K])
    subs["{{A_TILES_PER_SLICE}}"] = str(cfg.mapping.num_tiles_per_workgroup[DIM_M])
    subs["{{B_TILES_PER_SLICE}}"] = str(cfg.mapping.num_tiles_per_workgroup[DIM_N])
    subs["{{NUM_WAVES}}"] = str(cfg.num_waves)
    # 2-D cooperative split: (waves_m, waves_k) for A, (waves_n, waves_k) for B
    a_wm, a_wk, a_cm, a_ck = cfg.coop_a_split
    b_wn, b_wk, b_cn, b_ck = cfg.coop_b_split
    subs["{{COOP_A_WAVES_M}}"] = str(a_wm)
    subs["{{COOP_A_WAVES_K}}"] = str(a_wk)
    subs["{{COOP_A_M}}"] = str(a_cm)
    subs["{{COOP_A_K}}"] = str(a_ck)
    subs["{{MAX_COOP_A_M_START}}"] = str(
        max(0, cfg.mapping.num_tiles_per_workgroup[DIM_M] - a_cm)
    )
    subs["{{MAX_COOP_A_K_START}}"] = str(
        max(0, cfg.mapping.num_tiles_per_wave[DIM_K] - a_ck)
    )
    subs["{{COOP_B_WAVES_N}}"] = str(b_wn)
    subs["{{COOP_B_WAVES_K}}"] = str(b_wk)
    subs["{{COOP_B_N}}"] = str(b_cn)
    subs["{{COOP_B_K}}"] = str(b_ck)
    subs["{{MAX_COOP_B_N_START}}"] = str(
        max(0, cfg.mapping.num_tiles_per_workgroup[DIM_N] - b_cn)
    )
    subs["{{MAX_COOP_B_K_START}}"] = str(
        max(0, cfg.mapping.num_tiles_per_wave[DIM_K] - b_ck)
    )
    # Preshuffle layout parameters (f16: BK=32, 64 lanes, 16 bytes/lane).
    subs["{{STRIDE_N0_BYTES}}"] = str((gs[DIM_K] // 32) * 1024)
    subs["{{STRIDE_M0_BYTES}}"] = str((gs[DIM_K] // 32) * 1024)  # same formula as N
    subs["{{N_BLOCKS}}"] = str(gs[DIM_N] // 16)
    subs["{{M_BLOCKS}}"] = str(gs[DIM_M] // 16)
    subs["{{K_BLOCKS}}"] = str(gs[DIM_K] // 32)
    return subs


def compile_gemm(
    cfg,
    output_hsaco_path,
    print_ir_after_all=False,
    print_asm=False,
    num_vgprs=256,
    num_agprs=256,
):
    """Compile a GEMM config to HSACO.

    Returns (hsaco_path, asm_str). Handles b_path (lds/direct) and load_type
    (flat/buffer) via cfg fields. All compilation options (unroll, peeling, ll_sched,
    hoist_wait) are read from cfg.
    """
    from aster import ir
    from aster.compiler.core import compile_mlir_file_to_asm, assemble_to_hsaco

    subs = _make_substitutions(cfg)

    def preprocess(content):
        for pattern, replacement in subs.items():
            content = content.replace(pattern, replacement)
        return content

    mlir_key = (cfg.b_path, cfg.load_type)
    mlir_file = MLIR_FILES[mlir_key]
    lib_paths = get_kittens_16x16_lds_library_paths(use_buffer=cfg.use_buffer)

    pipeline = make_default_pass_pipeline(
        num_vgprs=num_vgprs,
        num_agprs=num_agprs,
        unroll_factor_multiplier=getattr(cfg, "unroll_factor_multiplier", 1),
        epilogue_peeling=getattr(cfg, "epilogue_peeling", True),
        ll_sched=getattr(cfg, "ll_sched", False),
        hoist_iter_arg_waits=getattr(cfg, "hoist_wait", False),
    )

    ctx = ir.Context()
    ctx.__enter__()
    try:
        from aster.compiler.core import PrintOptions

        asm, _ = compile_mlir_file_to_asm(
            get_mlir_file(mlir_file),
            cfg.kernel_name,
            pipeline,
            ctx,
            library_paths=lib_paths,
            preprocess=preprocess,
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=print_ir_after_all,
                print_asm=print_asm,
            ),
        )
        path = assemble_to_hsaco(
            asm,
            target=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            output_path=output_hsaco_path,
        )
        assert path is not None, "assemble_to_hsaco returned None"
        return path, asm
    finally:
        ctx.__exit__(None, None, None)


def execute_gemm_hsaco(cfg, hsaco_path, num_iterations, A, B, skip_gpu_check=False):
    """Execute a pre-compiled HSACO for a GEMM config.

    Returns (C_output, times_ns).

    Automatically preshuffles B when cfg.direct_b is True. Callers pass the original
    row-major B -- the preshuffle is applied here so there is a single code path for
    both test and bench.

    Skips (pytest.skip) if target GPU unavailable.
    """
    from aster.execution.core import execute_hsaco, InputArray, OutputArray
    from aster.execution.utils import system_has_gpu

    if not skip_gpu_check and not system_has_gpu(MCPU):
        pytest.skip(f"GPU {MCPU} not available, skip execution")

    # Preshuffle B for direct_b (single point of truth).
    A_gpu = shuffle_weight(A) if cfg.direct_a else A
    B_gpu = shuffle_weight(B) if cfg.direct_b else B

    import math

    C_output = np.zeros(math.prod(cfg.spec.operand_shape(OP_C)), dtype=np.float32)

    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=cfg.kernel_name,
        arguments=[
            InputArray(A_gpu.flatten()),
            InputArray(B_gpu.flatten()),
            OutputArray(C_output),
        ],
        grid_dim=(cfg.num_workgroups, 1, 1),
        block_dim=(cfg.num_threads, 1, 1),
        num_iterations=num_iterations,
    )
    return C_output, times_ns


class TestWeakScaleCorrectness:
    """Correctness gate: must pass before perf sweep runs."""

    # Problem sizes > 4000 in each dimension.
    # M = num_workgroups_per_kernel[M] * num_tiles_per_wg[M] * 16 (similarly N).
    # num_workgroups_per_kernel[N] must be power of 2 (delinearize from 1-D block ID).
    @pytest.mark.parametrize(
        "num_workgroups_per_kernel,num_waves_per_wg,num_tiles_per_wg",
        [
            # Divisible: tiles_per_wg % num_waves == 0
            ([32, 32, 1], [2, 2, 1], [4, 4, 1]),  # 2x2 waves, 2x2 tiles/wave
            ([16, 16, 1], [2, 2, 1], [8, 8, 1]),  # 2x2 waves, 4x4 tiles/wave
            ([32, 64, 1], [2, 2, 1], [4, 4, 1]),
            # OOB: tiles_per_wg % num_waves != 0 (excess waves load tile-0 region)
            # 4 waves, 2 tiles -> coop_a=1, 2 waves OOB
            ([64, 64, 1], [2, 2, 1], [2, 2, 1]),
            # 4 waves, 6 A-tiles -> coop_a=2, wave3 OOB
            ([32, 32, 1], [2, 2, 1], [6, 4, 1]),
            # 4 waves, 6 B-tiles -> coop_b=2, wave3 OOB
            ([32, 32, 1], [2, 2, 1], [4, 6, 1]),
            # 4 waves, 6x6 -> both OOB
            ([32, 32, 1], [2, 2, 1], [6, 6, 1]),
            # 8 waves, 4 tiles -> coop=1, 4 waves OOB
            ([32, 64, 1], [2, 4, 1], [4, 4, 1]),
        ],
        ids=[
            "div_2kx2k_twg4_w2x2",
            "div_2kx2k_twg8_w2x2",
            "div_2kx4k_twg4_w2x2",
            "oob_2kx2k_twg2_w2x2",
            "oob_6x4_twg6x4_w2x2",
            "oob_4x6_twg4x6_w2x2",
            "oob_6x6_twg6x6_w2x2",
            "oob_2kx4k_twg4_w2x4",
        ],
    )
    @pytest.mark.parametrize("a_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize("load_type", ["flat", "buffer"], ids=["flat", "buffer"])
    @pytest.mark.parametrize("b_path", ["lds", "direct_b"], ids=["lds", "direct_b"])
    def test_correctness(
        self,
        num_workgroups_per_kernel,
        num_waves_per_wg,
        num_tiles_per_wg,
        a_stages,
        load_type,
        b_path,
    ):
        """Constexpr GEMM verified against numpy."""
        if (b_path, load_type) not in MLIR_FILES:
            pytest.skip(f"({b_path}, {load_type}) not yet implemented")
        cfg = _make_weak_scaled_mapped_gemm_instance(
            num_workgroups_per_kernel,
            num_waves_per_wg,
            num_tiles_per_wg,
            k=128,
            a_stages=a_stages,
            load_type=load_type,
            b_path=b_path,
        )
        # Per-wave tile product > 16 requires too many registers.
        tpw = cfg.mapping.num_tiles_per_wave
        if tpw[DIM_M] * tpw[DIM_N] > 16:
            pytest.skip(
                f"per-wave tiles {tpw[DIM_M]}x{tpw[DIM_N]} product > 16 for {cfg.label}"
            )
        # Avoid unfeasible LDS sizes
        if cfg.lds_bytes >= LDS_SIZE:
            pytest.skip(f"LDS {cfg.lds_bytes} >= {LDS_SIZE}")
        np.random.seed(42)
        A = (np.random.randn(*cfg.spec.operand_shape(OP_A)) * 0.1).astype(np.float16)
        B = (np.random.randn(*cfg.spec.operand_shape(OP_B)) * 0.1).astype(np.float16)
        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            compile_gemm(cfg, tmp.name)
            C_output, _ = execute_gemm_hsaco(cfg, tmp.name, 1, A, B)

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestWeakScaledMappedGemmInstanceSerde:
    """Round-trip: WeakScaledMappedGemmInstance -> label -> from_label -> label."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(),
            dict(load_type="buffer"),
            dict(b_path="direct_b"),
            dict(load_type="buffer", b_path="direct_b"),
            dict(b_path="direct_ab"),
            dict(load_type="buffer", b_path="direct_ab"),
            dict(num_wg_per_cu=2, num_workgroups_per_kernel=[38, 16, 1]),
            dict(num_wg_per_cu=4, num_workgroups_per_kernel=[76, 16, 1]),
            dict(lcm_unroll=False),
            dict(unroll_factor_multiplier=3),
            dict(epilogue_peeling=False),
            dict(ll_sched=True),
            dict(hoist_wait=True),
            dict(pipeline_strategy=0),
            dict(pipeline_strategy=5),
            dict(pipeline_strategy=9),
            dict(
                lcm_unroll=False,
                unroll_factor_multiplier=2,
                epilogue_peeling=False,
                ll_sched=True,
                hoist_wait=True,
                num_wg_per_cu=2,
                num_workgroups_per_kernel=[38, 16, 1],
                load_type="buffer",
                b_path="direct_b",
                pipeline_strategy=7,
            ),
        ],
        ids=[
            "defaults",
            "buffer",
            "direct_b",
            "buffer_direct_b",
            "direct_ab",
            "buffer_direct_ab",
            "wgcu2",
            "wgcu4",
            "nolcm",
            "um3",
            "nopeel",
            "llsched",
            "hoistwait",
            "ps0",
            "ps5",
            "ps9",
            "all_flags",
        ],
    )
    def test_label_roundtrip(self, kwargs):
        base = dict(
            num_workgroups_per_kernel=[19, 16, 1],
            num_waves_per_wg=[2, 2, 1],
            num_tiles_per_wg=[8, 8, 2],
            a_stages=2,
            k=8192,
            pipeline_strategy=1,
        )
        base.update(kwargs)
        cfg = _make_weak_scaled_mapped_gemm_instance(**base)
        restored = WeakScaledMappedGemmInstance.from_label(cfg.label)
        assert restored.label == cfg.label
        for field in [
            "gemm_size",
            "a_stages",
            "b_stages",
            "pipeline_strategy",
            "load_type",
            "b_path",
            "num_wg_per_cu",
            "lcm_unroll",
            "unroll_factor_multiplier",
            "epilogue_peeling",
            "ll_sched",
            "hoist_wait",
            "num_workgroups_per_kernel",
            "num_waves_per_workgroup",
            "num_tiles_per_workgroup",
            "num_threads",
        ]:
            assert getattr(restored, field) == getattr(
                cfg, field
            ), f"{field}: {getattr(restored, field)} != {getattr(cfg, field)}"

    def test_from_label_rejects_garbage(self):
        with pytest.raises(ValueError, match="Cannot parse label"):
            WeakScaledMappedGemmInstance.from_label("not_a_valid_label")

    def test_from_label_rejects_truncated(self):
        cfg = _make_weak_scaled_mapped_gemm_instance(
            [19, 16, 1], [2, 2, 1], [8, 8, 1], k=4096, a_stages=2, pipeline_strategy=1
        )
        with pytest.raises(ValueError):
            WeakScaledMappedGemmInstance.from_label(cfg.label[:-5])


if __name__ == "__main__":
    raise SystemExit(
        "Use bench/bench_perf_001_gemm_fp16_weak_scaled.py <label> for single-config runs."
    )
