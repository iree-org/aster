"""Standalone profiling runner for kittens kernels.

Usage:
    python run_profiling.py --list                    # list available tests
    python run_profiling.py --test gemm_fp16_4wave_k128  # run single test
    python run_profiling.py --num-blocks 304          # full MI300 occupancy
"""

import argparse

import pytest

# Import run_config and mutate the SAME object so run_kittens_kernel sees changes.
from kittens_helpers import run_config

# Short aliases keep ALL_TESTS entries on single lines.
from test_001_gemm_fp16_lds_1buf import TestKittensGEMMLDS1Buffer as LDS1B
from test_002_gemm_fp16_lds_2buf import TestKittensGEMMLDS2Buffer as LDS2B
from test_003_gemm_fp16_lds_3buf import TestKittensGEMMLDS3Buffer as LDS3B
from test_004_gemm_fp16_2wave_lds import TestKittensGEMM2WaveLDS as GEMM2WLDS
from test_005_gemm_fp16_lds_pipelined import TestKittensGEMMLDSPipelined as LDSP
from test_006_gemm_fp16_multitile_lds_pipelined import (
    TestKittensGEMMMultiTileLDSPipelined as MTLDSP,
)
from test_007_gemm_fp16_4wave_lds_pipelined import (
    TestKittensGEMM4WaveLDSPipelined as G4WLDSP,
)
from test_perf_001_gemm_fp16_weak_scaled import TestWeakScaleCorrectness as WSCorr
from bench.bench_perf_sweep_001_gemm_fp16_weak_scaled import (
    bench_perf_sweep as _bench_perf_sweep,
)

# Registry: (kernel_name, test_fn, args, kwargs)
# fmt: off
ALL_TESTS = [
    ("gemm_fp16_lds_1buf_k32",                   LDS1B().test_gemm_lds_1buf,                      [], {"k": 32}),
    ("gemm_fp16_lds_1buf_k64",                   LDS1B().test_gemm_lds_1buf,                      [], {"k": 64}),
    ("gemm_fp16_lds_1buf_k128",                  LDS1B().test_gemm_lds_1buf,                      [], {"k": 128}),
    ("gemm_fp16_lds_2buf_k32",                   LDS2B().test_gemm_lds_2buf,                      [], {"k": 32}),
    ("gemm_fp16_lds_2buf_k64",                   LDS2B().test_gemm_lds_2buf,                      [], {"k": 64}),
    ("gemm_fp16_lds_2buf_k128",                  LDS2B().test_gemm_lds_2buf,                      [], {"k": 128}),
    ("gemm_fp16_lds_3buf_k48",                   LDS3B().test_gemm_lds_3buf,                      [], {"k": 48}),
    ("gemm_fp16_lds_3buf_k64",                   LDS3B().test_gemm_lds_3buf,                      [], {"k": 64}),
    ("gemm_fp16_lds_3buf_k96",                   LDS3B().test_gemm_lds_3buf,                      [], {"k": 96}),
    ("gemm_fp16_2wave_lds_k32",                  GEMM2WLDS().test_gemm_2wave_lds,                 [], {"k": 32}),
    ("gemm_fp16_2wave_lds_k64",                  GEMM2WLDS().test_gemm_2wave_lds,                 [], {"k": 64}),
    ("gemm_fp16_2wave_lds_k128",                 GEMM2WLDS().test_gemm_2wave_lds,                 [], {"k": 128}),
    ("gemm_fp16_mt_lds_pipe_2s_k64",             MTLDSP().test_gemm_multitile_lds_pipelined,      [], {"k": 64, "num_stages": 2}),
    ("gemm_fp16_mt_lds_pipe_2s_k128",            MTLDSP().test_gemm_multitile_lds_pipelined,      [], {"k": 128, "num_stages": 2}),
    ("gemm_fp16_mt_lds_pipe_3s_k64",             MTLDSP().test_gemm_multitile_lds_pipelined,      [], {"k": 64, "num_stages": 3}),
    ("gemm_fp16_mt_lds_pipe_3s_k128",            MTLDSP().test_gemm_multitile_lds_pipelined,      [], {"k": 128, "num_stages": 3}),
    ("gemm_fp16_lds_pipe_2s_k64",                LDSP().test_gemm_lds_pipelined,                   [], {"k": 64, "num_stages": 2}),
    ("gemm_fp16_lds_pipe_2s_k128",               LDSP().test_gemm_lds_pipelined,                   [], {"k": 128, "num_stages": 2}),
    ("gemm_fp16_lds_pipe_3s_k64",                LDSP().test_gemm_lds_pipelined,                   [], {"k": 64, "num_stages": 3}),
    ("gemm_fp16_lds_pipe_3s_k128",               LDSP().test_gemm_lds_pipelined,                   [], {"k": 128, "num_stages": 3}),
    ("gemm_fp16_4wave_lds_pipe_2s_k64",          G4WLDSP().test_gemm_4wave_lds_pipelined,          [], {"k": 64, "num_stages": 2}),
    ("gemm_fp16_4wave_lds_pipe_2s_k128",         G4WLDSP().test_gemm_4wave_lds_pipelined,          [], {"k": 128, "num_stages": 2}),
    ("gemm_fp16_4wave_lds_pipe_3s_k64",          G4WLDSP().test_gemm_4wave_lds_pipelined,          [], {"k": 64, "num_stages": 3}),
    ("gemm_fp16_4wave_lds_pipe_3s_k128",         G4WLDSP().test_gemm_4wave_lds_pipelined,          [], {"k": 128, "num_stages": 3}),
    # Weak-scaling correctness (representative configs)
    ("ws_correctness_2x2_2s",                    WSCorr().test_correctness,                       [], {"num_workgroups_per_kernel": [1, 1, 1], "num_waves_per_wg": [1, 1, 1], "num_tiles_per_wg": [2, 2, 1], "num_stages": 2}),
    ("ws_correctness_4x4_3s",                    WSCorr().test_correctness,                       [], {"num_workgroups_per_kernel": [1, 1, 1], "num_waves_per_wg": [2, 2, 1], "num_tiles_per_wg": [4, 4, 1], "num_stages": 3}),
    # Weak-scaling perf sweep (runs all 24 configs)
    ("ws_perf_sweep",                            _bench_perf_sweep,                               [], {}),
]
# fmt: on

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run kittens tests")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Number of workgroups (default: 1, use 304 for full MI300 occupancy)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of kernel launches per test (default: 1)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run only tests whose name contains this substring (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test names and exit",
    )
    cli_args = parser.parse_args()

    if cli_args.list:
        for name, _, _, _ in ALL_TESTS:
            print(name)
        raise SystemExit(0)

    # Mutate the shared run_config object so run_kittens_kernel sees the changes.
    run_config.num_blocks = cli_args.num_blocks
    run_config.num_iterations = cli_args.num_iterations

    tests = ALL_TESTS
    if cli_args.test:
        tests = [(n, f, a, kw) for n, f, a, kw in ALL_TESTS if cli_args.test in n]
        if not tests:
            print(f"No tests matching '{cli_args.test}'. Available:")
            for name, _, _, _ in ALL_TESTS:
                print(f"  {name}")
            raise SystemExit(1)

    for name, test_fn, args, kwargs in tests:
        try:
            test_fn(*args, **kwargs)
        except pytest.skip.Exception as e:
            print(f"  SKIPPED: {e}")
