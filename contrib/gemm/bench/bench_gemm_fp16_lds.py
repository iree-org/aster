"""Benchmark: TFLOPS sweep for gemm_fp16_lds (multi-CU GEMM with LDS tiling).

Phase 1: Parallel compilation (MLIR -> HSACO) across all configs.
Phase 2: Parallel GPU execution with round-robin across available GPUs,
         each config in its own subprocess for crash isolation.

Sweep axes:
  num_waves x num_m_waves (wave grid) x m_tile_mult x n_tile_mult
  x k_t (k_tile / 32) x k_factor (K = k_factor * k_tile) x swizzle.

Usage (sweep):
    python .../bench_gemm_fp16_lds.py --sweep
    python .../bench_gemm_fp16_lds.py --sweep --full-sweep
    python .../bench_gemm_fp16_lds.py --sweep --num-gpus 8 --compile-workers 16

Usage (single config):
    python .../bench_gemm_fp16_lds.py \\
        --m 4096 --n 4096 --k 4096 \\
        --m-tile 64 --n-tile 64 --k-tile 64 \\
        --num-waves 4 --num-m-waves 2

Usage (compile only / execute pre-compiled HSACO):
    ... --compile-only --hsaco /tmp/output.hsaco
    ... --hsaco /tmp/output.hsaco
"""

import itertools
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, ".."))  # contrib/gemm/ for gemm_fp16_lds
sys.path.insert(0, os.path.join(_DIR, "..", "..", "..", ".."))  # repo root
sys.path.insert(0, os.path.join(_DIR, "..", "..", "kittens", "test", "bench"))

from dataclasses import dataclass

import numpy as np

from gemm_fp16_lds import (
    GEMMConfig,
    MCPU,
    WAVEFRONT_SIZE,
    _get_library_paths,
    _make_substitutions,
)
from bench_harness import (
    NUM_ITERATIONS,
    add_single_cli_args,
    add_sweep_cli_args,
    bench_perf_sweep,
    make_sweep_filter,
    run_single,
)

KERNEL_NAME = "gemm_fp16_lds"
_MLIR_FILE = os.path.join(_DIR, "..", "gemm_fp16_lds.mlir")

# ---- Sweep parameters ----------------------------------------------------

# Tile sizes are expressed as multiples of the minimum: 16 * num_waves.
# This guarantees m_tile % (16 * num_waves) == 0 (and same for n_tile).
_M_TILE_MULTS = [1, 2, 3, 4]
_N_TILE_MULTS = [1, 2, 4]
# k_t = k_tile // 32 (k_tile must be a multiple of 32).
_K_T_VALUES = [2, 4]
# K = k_factor * k_tile.
_K_FACTORS = [32, 64, 128]
# Wave counts to sweep.
_NUM_WAVES_CONFIGS = [2, 4, 8]
# Base workgroup counts along M and N. n_wg scaled by N_WG_MULTS.
_M_WG_BASE = 38
_N_WG_BASE = 16
_N_WG_MULTS = [1, 2, 4]

MIN_DIM = 512  # Skip configs where M, N, or K < this.


# ---- BenchConfig ---------------------------------------------------------


@dataclass
class BenchConfig(GEMMConfig):
    """GEMMConfig extended with bench harness interface properties."""

    num_wg_per_cu: int = 1
    _label_suffix: str = ""

    @property
    def label(self) -> str:
        swizzle_str = f"_sw{self.swizzle}" if self.swizzle > 1 else ""
        return (
            f"m{self.m}xn{self.n}xk{self.k}"
            f"_mt{self.m_tile}xnt{self.n_tile}_kt{self.k_t}"
            f"_nw{self.num_waves}_mw{self.num_m_waves}"
            f"{swizzle_str}"
            f"{self._label_suffix}"
        )

    @property
    def kernel_name(self) -> str:
        return KERNEL_NAME

    @property
    def num_workgroups(self) -> int:
        return self.num_blocks

    @property
    def m_dim(self) -> int:
        return self.m

    @property
    def n_dim(self) -> int:
        return self.n

    @property
    def total_flops(self) -> int:
        return 2 * self.m * self.n * self.k

    @property
    def estimated_agprs(self) -> int:
        """4 AGPRs per 16x16 output tile; each wave owns m_t_pw_c x n_t_pw_c tiles."""
        return self.m_t_pw_c * self.n_t_pw_c * 4

    @property
    def estimated_vgprs(self) -> int:
        """Global-load buffers + LDS-read buffers + overhead.

        Each 16x32 f16 transfer tile occupies 4 VGPRs per lane (dwordx4).
        """
        a_load = self.m_t_per_wave * self.k_t * 4
        b_load = self.n_t_per_wave * self.k_t * 4
        a_lds = self.m_t_pw_c * self.k_t * 4
        b_lds = self.n_t_pw_c * self.k_t * 4
        return a_load + b_load + a_lds + b_lds + 20  # 20 for addresses/loop vars

    @property
    def lds_bytes(self) -> int:
        return self.a_lds_bytes + self.b_lds_bytes


# ---- Compile / execute ---------------------------------------------------


def compile_gemm(
    cfg: BenchConfig,
    output_hsaco_path: str,
    print_ir_after_all: bool = False,
    print_asm: bool = False,
    num_vgprs: int = 256,
    num_agprs: int = 256,
):
    """Compile a BenchConfig to an HSACO file.

    Returns (hsaco_path, asm_str).
    """
    from aster import ir
    from aster.pass_pipelines import make_default_pass_pipeline
    from aster.compiler.core import (
        compile_mlir_file_to_asm,
        PrintOptions,
        assemble_to_hsaco,
    )

    subs = _make_substitutions(cfg)

    def preprocess(content: str) -> str:
        for pattern, replacement in subs.items():
            content = content.replace(pattern, replacement)
        return content

    pipeline = make_default_pass_pipeline(num_vgprs=num_vgprs, num_agprs=num_agprs)
    lib_paths = _get_library_paths()

    ctx = ir.Context()
    ctx.__enter__()
    try:
        asm, _ = compile_mlir_file_to_asm(
            _MLIR_FILE,
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


def execute_gemm_hsaco(
    cfg: BenchConfig,
    hsaco_path: str,
    num_iterations: int,
    A: np.ndarray,
    B: np.ndarray,
    skip_gpu_check: bool = False,
):
    """Execute a pre-compiled HSACO for a BenchConfig.

    Returns (C_output, times_ns).
    """
    from aster.execution.core import execute_hsaco
    from aster.execution.utils import system_has_gpu

    if not skip_gpu_check and not system_has_gpu(MCPU):
        import pytest

        pytest.skip(f"GPU {MCPU} not available, skipping execution")

    C_output = np.zeros(cfg.m_dim * cfg.n_dim, dtype=np.float32)
    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=cfg.kernel_name,
        input_arrays=[A.flatten(), B.flatten()],
        output_arrays=[C_output],
        grid_dim=(cfg.num_workgroups, 1, 1),
        block_dim=(cfg.num_threads, 1, 1),
        num_iterations=num_iterations,
    )
    return C_output, times_ns


# ---- Sweep ---------------------------------------------------------------


def _wave_grids(num_waves: int):
    """Return all valid num_m_waves values for num_waves (every divisor)."""
    return [mw for mw in range(1, num_waves + 1) if num_waves % mw == 0]


def _generate_configs(
    sample_size: int = 2000, check_regs: bool = True, sweep_filter=None
):
    """Generate the sweep grid, pre-filter by register and LDS budget."""
    from aster.compiler.metadata import compute_register_budget

    configs = []
    filtered = []

    for num_waves in _NUM_WAVES_CONFIGS:
        for num_m_waves in _wave_grids(num_waves):
            for m_mult, n_mult in itertools.product(_M_TILE_MULTS, _N_TILE_MULTS):
                m_tile = m_mult * num_waves * 16
                n_tile = n_mult * num_waves * 16
                for k_t in _K_T_VALUES:
                    k_tile = k_t * 32
                    for k_factor in _K_FACTORS:
                        k = k_factor * k_tile
                        for n_wg_mult in _N_WG_MULTS:
                            n_wg = _N_WG_BASE * n_wg_mult
                            m_wg = _M_WG_BASE
                            m = m_wg * m_tile
                            n = n_wg * n_tile
                            if m < MIN_DIM or n < MIN_DIM or k < MIN_DIM:
                                continue
                            try:
                                cfg = BenchConfig(
                                    m,
                                    n,
                                    k,
                                    m_tile,
                                    n_tile,
                                    k_tile,
                                    num_waves,
                                    swizzle=1,
                                    num_m_waves=num_m_waves,
                                )
                            except AssertionError:
                                continue
                            if check_regs:
                                max_v, max_a, lds_per_wg = compute_register_budget(
                                    cfg.num_threads,
                                    mcpu=MCPU,
                                    num_wg_per_cu=cfg.num_wg_per_cu,
                                )
                                if cfg.lds_bytes > lds_per_wg:
                                    filtered.append(
                                        (
                                            cfg.label,
                                            f"LDS {cfg.lds_bytes} > {lds_per_wg}",
                                        )
                                    )
                                    continue
                                if cfg.estimated_vgprs > max_v:
                                    filtered.append(
                                        (
                                            cfg.label,
                                            f"est_vgpr {cfg.estimated_vgprs} > {max_v}",
                                        )
                                    )
                                    continue
                                if cfg.estimated_agprs > max_a:
                                    filtered.append(
                                        (
                                            cfg.label,
                                            f"est_agpr {cfg.estimated_agprs} > {max_a}",
                                        )
                                    )
                                    continue
                            configs.append(cfg)

    if filtered:
        import tempfile as _tmp

        fd, filt_path = _tmp.mkstemp(
            prefix="bench_filtered_", suffix=".txt", dir="/tmp"
        )
        with os.fdopen(fd, "w") as f:
            for label, reason in filtered:
                f.write(f"{label}: {reason}\n")
        print(
            f"{len(filtered)} configs skipped by pre-compile filter "
            f"(details in {filt_path})"
        )

    if sweep_filter:
        before = len(configs)
        configs = [c for c in configs if sweep_filter(c)]
        print(f"Sweep filter: {before} -> {len(configs)} eligible configs")

    import random

    total = len(configs)
    n = min(sample_size, total) if sample_size > 0 else total
    if n < total:
        configs = random.sample(configs, n)
    print(f"Compiling {n} / {total} eligible configs")
    return configs


def _repro_cmd(cfg: BenchConfig, num_iterations: int) -> str:
    swizzle_str = f" --swizzle {cfg.swizzle}" if cfg.swizzle > 1 else ""
    return (
        f"python contrib/gemm/bench/bench_gemm_fp16_lds.py"
        f" --m {cfg.m} --n {cfg.n} --k {cfg.k}"
        f" --m-tile {cfg.m_tile} --n-tile {cfg.n_tile} --k-tile {cfg.k_tile}"
        f" --num-waves {cfg.num_waves} --num-m-waves {cfg.num_m_waves}"
        f"{swizzle_str}"
        f" --iterations {num_iterations}"
    )


def _compile_fn(cfg: BenchConfig, output_hsaco_path: str, **kwargs):
    return compile_gemm(cfg, output_hsaco_path, **kwargs)


# ---- Entry point ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="gemm_fp16_lds benchmark: sweep or single-config repro",
    )
    add_sweep_cli_args(parser)

    # Single-config positional args.
    parser.add_argument("--m", type=int, help="Total M dimension")
    parser.add_argument("--n", type=int, help="Total N dimension")
    parser.add_argument("--k", type=int, help="Total K dimension")
    parser.add_argument("--m-tile", type=int, help="M tile size per workgroup")
    parser.add_argument("--n-tile", type=int, help="N tile size per workgroup")
    parser.add_argument(
        "--k-tile", type=int, help="K tile size (must be multiple of 32)"
    )
    parser.add_argument("--num-waves", type=int, help="Wavefronts per workgroup")
    parser.add_argument(
        "--num-m-waves",
        type=int,
        default=0,
        help="Waves in M direction (0 = 1D mode = num_waves)",
    )
    parser.add_argument("--swizzle", type=int, default=1, help="CTA swizzle factor")
    add_single_cli_args(parser)

    # Sweep-dimension pins (filter the sweep grid to a specific value).
    parser.add_argument("--filter-num-waves", type=int, default=None)
    parser.add_argument("--filter-num-m-waves", type=int, default=None)
    parser.add_argument("--filter-k", type=int, default=None)

    args = parser.parse_args()

    if args.sweep or args.full_sweep:
        _SWEEP_ATTR_MAP = {
            "filter_num_waves": "num_waves",
            "filter_num_m_waves": "num_m_waves",
            "filter_k": "k",
        }
        sweep_filter = make_sweep_filter(args, _SWEEP_ATTR_MAP)
        all_configs = _generate_configs(
            sample_size=getattr(args, "compile_sample", 2000),
            check_regs=not getattr(args, "no_reg_filter", False),
            sweep_filter=sweep_filter,
        )
        bench_perf_sweep(
            configs=all_configs,
            compile_fn=_compile_fn,
            repro_cmd_fn=_repro_cmd,
            full_sweep=args.full_sweep,
            num_gpus=args.num_gpus,
            compile_workers=args.compile_workers,
            compile_timeout=getattr(args, "compile_timeout", 60),
        )
    else:
        required = ["m", "n", "k", "m_tile", "n_tile", "k_tile", "num_waves"]
        missing = [a for a in required if getattr(args, a, None) is None]
        if missing:
            flags = ", ".join(f"--{a.replace('_', '-')}" for a in missing)
            parser.error(f"Single-config mode requires: {flags}")

        cfg = BenchConfig(
            args.m,
            args.n,
            args.k,
            args.m_tile,
            args.n_tile,
            args.k_tile,
            args.num_waves,
            args.swizzle,
            args.num_m_waves,
        )
        print(
            f"Config: {cfg.label}"
            f"  grid={cfg.m_wg}x{cfg.n_wg}  blocks={cfg.num_blocks}"
            f"  threads={cfg.num_threads}"
        )
        run_single(cfg, compile_gemm, args, execute_fn=execute_gemm_hsaco)
