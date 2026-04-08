"""HipBLAS GEMM benchmark — Python wrapper around the hipblaslt-bench CLI.

Example equivalent shell command:
    hipblaslt-bench \\
        -m 8192 -n 8192 -k 8192 \\
        --a_type f16_r --b_type f16_r --c_type f32_r --d_type f32_r \\
        --compute_type f32_r \\
        --algo_method all \\
        --transB T
"""

import re
import subprocess
from typing import Any

from common.gemm_config import DTYPE_TO_ROCBLAS, GEMMConfig

# hipblaslt-bench emits a "Winner:" line followed by a CSV header line of the
# form "[N]:transA,..." and a data row beneath it.
# Relevant columns are "us" (microseconds) or "hipblaslt-Gflops".
_HEADER_RE = re.compile(r"^\[\d+\]:transA,", re.IGNORECASE)
_US_HEADER = "us"
_GFLOPS_HEADER = "hipblaslt-Gflops"


def _parse_output(stdout: str, flops: int) -> dict[str, Any]:
    """Extract timing from the winner section of hipblaslt-bench output."""
    lines = stdout.splitlines()
    best: dict[str, Any] = {}
    header: list[str] = []
    found = False

    for line in lines:
        stripped = line.strip()
        if not found:
            if stripped.startswith("Winner"):
                found = True
            continue
        if _HEADER_RE.match(stripped):
            # Strip the "[N]:" prefix to get the raw CSV header fields.
            header_csv = stripped.split(":", 1)[1]
            header = [h.strip() for h in header_csv.split(",")]
            continue
        if header and stripped and not stripped.startswith("#"):
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            us_str = row.get(_US_HEADER)
            gflops_str = row.get(_GFLOPS_HEADER)
            ms: float | None = None
            tflops: float | None = None
            if us_str is not None:
                try:
                    ms = float(us_str) / 1_000.0
                    tflops = flops / (ms * 1e9)
                except ValueError:
                    continue
            elif gflops_str is not None:
                try:
                    gflops = float(gflops_str)
                    tflops = gflops / 1_000.0
                    ms = flops / (tflops * 1e9)
                except ValueError:
                    continue
            if ms is not None and tflops is not None:
                if not best or ms < best["ms"]:
                    best = {"ms": ms, "tflops": tflops}

    if not best:
        raise RuntimeError(
            "Could not parse hipblaslt-bench output. Raw output:\n" + stdout
        )
    return best


class HipBLASBenchmark:
    """Profiles hipBLAS GEMM by invoking the hipblaslt-bench CLI."""

    def __init__(self, config: GEMMConfig, executable: str = "hipblaslt-bench") -> None:
        self.config = config
        self.executable = executable

    def run(self, warmup: int = 5, iters: int = 10) -> dict:
        """Invoke hipblaslt-bench and return timing results for the best algorithm.

        Returns:
            {"ms": float, "tflops": float}
        """
        cfg = self.config
        dtype_str = DTYPE_TO_ROCBLAS.get(cfg.dtype, "f16_r")
        trans_b = "T" if cfg.transpose_b else "N"

        cmd = [
            self.executable,
            "-m",
            str(cfg.m),
            "-n",
            str(cfg.n),
            "-k",
            str(cfg.k),
            "--a_type",
            dtype_str,
            "--b_type",
            dtype_str,
            "--c_type",
            "f32_r",
            "--d_type",
            "f32_r",
            "--compute_type",
            "f32_r",
            "--algo_method",
            "all",
            "--transB",
            trans_b,
            "--cold_iters",
            str(warmup),
            "--iters",
            str(iters),
        ]

        print("Running:", " ".join(cmd), flush=True)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        # Only print the Winner section (suppress per-solution output).
        in_winner = False
        for line in result.stdout.splitlines():
            if line.strip().startswith("Winner"):
                in_winner = True
            if in_winner:
                print(line)
        return _parse_output(result.stdout, cfg.flops())
