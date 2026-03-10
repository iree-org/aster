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
from test_000_zero_c import TestKittensZeroC as ZeroC
from test_001_load_store_a import TestKittensLoadStoreA as LoadStore
from test_002_mfma import TestKittensMFMA as MFMA
from test_003_gemm_fp16_fixed import TestKittensGEMM as GEMM
from test_004_gemm_fp16_sched import TestKittensGEMMSched as GEMMSched
from test_005_gemm_fp16_loop import TestKittensGEMMLoop as GEMMLoop
from test_006_gemm_fp16_2wave import TestKittensGEMM2Wave as GEMM2W
from test_007_gemm_fp16_4wave import TestKittensGEMM4Wave as GEMM4W
from test_008_lds_roundtrip import TestKittensLDSRoundtrip as LDSRt
from test_009_gemm_fp16_lds_1buf import TestKittensGEMMLDS1Buffer as LDS1B
from test_010_gemm_fp16_lds_2buf import TestKittensGEMMLDS2Buffer as LDS2B
from test_011_gemm_fp16_lds_3buf import TestKittensGEMMLDS3Buffer as LDS3B
from test_012_lds_roundtrip_xor import TestKittensLDSRoundtripXorSwizzle as LDSRtXor
from test_013_gemm_fp16_2wave_lds import TestKittensGEMM2WaveLDS as GEMM2WLDS
from test_014_gemm_fp16_lds_pipelined import TestKittensGEMMLDSPipelined as LDSP
from test_015_gemm_fp16_multitile_direct import (
    TestKittensGEMMMultiTileDirect as MTDirect,
)
from test_016_gemm_fp16_multitile_lds_pipelined import (
    TestKittensGEMMMultiTileLDSPipelined as MTLDSP,
)
from test_018_gemm_fp16_4wave_lds_pipelined import (
    TestKittensGEMM4WaveLDSPipelined as G4WLDSP,
)
from test_perf_001_gemm_fp16_weak_scaled import TestWeakScaleCorrectness as WSCorr
from bench.bench_perf_sweep_001_gemm_fp16_weak_scaled import (
    bench_perf_sweep as _bench_perf_sweep,
)

# Registry: (kernel_name, test_fn, args, kwargs)
# fmt: off
ALL_TESTS = [
    ("test_zero_C",                              ZeroC().test_zero_C_produces_zeros,              [], {}),
    ("test_load_store_A",                        LoadStore().test_load_store_roundtrip,            [], {}),
    ("test_mfma",                                MFMA().test_mfma_matmul,                         [], {}),
    ("gemm_fp16_16x16x128",                      GEMM().test_gemm_16x16x128,                      [], {}),
    ("gemm_fp16_16x16x128_sched",                GEMMSched().test_gemm_16x16x128_sched,           [], {}),
    ("gemm_fp16_16x16xK_k128",                   GEMMLoop().test_gemm_16x16xK,                    [], {"k": 128}),
    ("gemm_fp16_16x16xK_k4096",                  GEMMLoop().test_gemm_16x16xK,                    [], {"k": 4096}),
    ("gemm_fp16_2wave_k32",                      GEMM2W().test_gemm_2wave,                        [], {"k": 32}),
    ("gemm_fp16_2wave_k64",                      GEMM2W().test_gemm_2wave,                        [], {"k": 64}),
    ("gemm_fp16_2wave_k128",                     GEMM2W().test_gemm_2wave,                        [], {"k": 128}),
    ("gemm_fp16_4wave_k32",                      GEMM4W().test_gemm_4wave,                        [], {"k": 32}),
    ("gemm_fp16_4wave_k64",                      GEMM4W().test_gemm_4wave,                        [], {"k": 64}),
    ("gemm_fp16_4wave_k128",                     GEMM4W().test_gemm_4wave,                        [], {"k": 128}),
    ("gemm_fp16_multitile_direct_k32",           MTDirect().test_gemm_multitile_direct,            [], {"k": 32}),
    ("gemm_fp16_multitile_direct_k64",           MTDirect().test_gemm_multitile_direct,            [], {"k": 64}),
    ("gemm_fp16_multitile_direct_k128",          MTDirect().test_gemm_multitile_direct,            [], {"k": 128}),
    ("lds_roundtrip",                            LDSRt().test_lds_roundtrip_f16,                  [], {}),
    ("lds_roundtrip_xor_swizzle",                LDSRtXor().test_lds_roundtrip_xor_swizzle_f16,   [], {}),
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
    ("ws_correctness_2x2_2s",                    WSCorr().test_correctness,                       [], {"m_wg": 1, "n_wg": 1, "m_waves": 1, "n_waves": 1, "m_tiles_wg": 2, "n_tiles_wg": 2, "num_stages": 2}),
    ("ws_correctness_4x4_3s",                    WSCorr().test_correctness,                       [], {"m_wg": 1, "n_wg": 1, "m_waves": 2, "n_waves": 2, "m_tiles_wg": 4, "n_tiles_wg": 4, "num_stages": 3}),
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
