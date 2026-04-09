"""RocBLAS GEMM benchmark — Python wrapper around the rocblas-bench CLI.

Example equivalent shell command:
    rocblas-bench -f gemm_ex \\
        --a_type f16_r --b_type f16_r --c_type f32_r --d_type f32_r \\
        --compute_type f32_r \\
        -m 8192 -n 8192 -k 8192 \\
        --transposeA N --transposeB T \\
        -j 5 -i 10
"""

import re
import shutil
import subprocess
from typing import Any

from common.gemm_config import DTYPE_TO_ROCBLAS, GEMMConfig

# rocblas-bench prints timing lines like:
#   transA,transB,M,N,K,...,rocblas-Gflops,us,...
# followed by a data row.  We look for "us" header and the corresponding
# column value.
_HEADER_RE = re.compile(r"^transA,")
_US_HEADER = "us"
_GFLOPS_HEADER = "rocblas-Gflops"


def _parse_output(stdout: str, flops: int) -> dict[str, Any]:
    """Extract timing from rocblas-bench output and return result dict."""
    lines = stdout.splitlines()
    header: list[str] = []
    for i, line in enumerate(lines):
        if _HEADER_RE.match(line.strip()):
            header = [h.strip() for h in line.split(",")]
            # The data row immediately follows the header.
            if i + 1 < len(lines):
                data = [d.strip() for d in lines[i + 1].split(",")]
                row = dict(zip(header, data))
                us_str = row.get(_US_HEADER)
                gflops_str = row.get(_GFLOPS_HEADER)
                if us_str is not None:
                    ms = float(us_str) / 1_000.0
                    tflops = flops / (ms * 1e9)
                    return {"ms": ms, "tflops": tflops}
                if gflops_str is not None:
                    gflops = float(gflops_str)
                    tflops = gflops / 1_000.0
                    ms = flops / (tflops * 1e9)
                    return {"ms": ms, "tflops": tflops}
            break

    raise RuntimeError("Could not parse rocblas-bench output. Raw output:\n" + stdout)


class RocBLASBenchmark:
    """Profiles rocBLAS GEMM by invoking the rocblas-bench CLI."""

    def __init__(self, config: GEMMConfig, executable: str = "rocblas-bench") -> None:
        self.config = config
        self.executable = executable

    def run(self, warmup: int = 5, iters: int = 10) -> dict:
        """Invoke rocblas-bench and return timing results.

        Returns:
            {"ms": float, "tflops": float}
        """
        if not shutil.which(self.executable):
            raise RuntimeError(
                f"Couldn't find {self.executable}, try adding it to the path or "
                f"specifying the tool path in the command line with:\n"
                f"`--rocblas-bench=<path-to-tool>`"
            )

        cfg = self.config
        dtype_str = DTYPE_TO_ROCBLAS.get(cfg.dtype, "f16_r")
        trans_b = "T" if cfg.transpose_b else "N"

        cmd = [
            self.executable,
            "-f",
            "gemm_ex",
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
            "-m",
            str(cfg.m),
            "-n",
            str(cfg.n),
            "-k",
            str(cfg.k),
            "--transposeA",
            "N",
            "--transposeB",
            trans_b,
            "-j",
            str(warmup),
            "-i",
            str(iters),
        ]

        print("Running:", " ".join(cmd), flush=True)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        return _parse_output(result.stdout, cfg.flops())
