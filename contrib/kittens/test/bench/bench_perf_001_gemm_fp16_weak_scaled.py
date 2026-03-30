"""Single-config GEMM benchmark: compile + execute one config by label.

Paste a label from sweep output to reproduce the exact same compilation and execution.

Usage:
    python .../bench_perf_001_gemm_fp16_weak_scaled.py \
        --label m4864xn4096xk8192_wg38x32_w2x2_twg8x8x1_s2_bs2_ps1_direct_b_flat

    ... --label <label> --compile-only --hsaco /tmp/output.hsaco
    ... --label <label> --hsaco /tmp/output.hsaco
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from test_perf_001_gemm_fp16_weak_scaled import (
    WeakScaleConfig,
    compile_gemm,
    execute_gemm_hsaco,
)
from bench_harness import add_single_cli_args, run_single

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-config GEMM benchmark (paste label from sweep output)",
    )
    parser.add_argument("label", type=str, help="Config label from sweep output")
    add_single_cli_args(parser)
    args = parser.parse_args()

    cfg = WeakScaleConfig.from_label(args.label)
    run_single(cfg, compile_gemm, args, execute_fn=execute_gemm_hsaco)
