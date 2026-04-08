"""Aiter GEMM benchmark — wraps aiter's TunedGemm API.

aiter operates on 16-bit inputs (fp16 or bf16); fp32 input is not
supported.
"""

import os
import sys
from pathlib import Path

import torch

from common.gemm_config import GEMMConfig
from common.profiling import profile_fn


def _resolve_rocm_lib() -> str | None:
    """Return the ROCm lib directory to add to LIBRARY_PATH, or None.

    Resolution order:
      1. ROCM_LIB_PATH env var — explicit per-session override.
      2. $ROCM_PATH/lib        — standard ROCm env var set by the ROCm installer.
      3. /opt/rocm/lib         — canonical versioned-independent symlink.
      4. /opt/rocm-7.2.1/lib   — versioned fallback.
    """
    explicit = os.environ.get("ROCM_LIB_PATH")
    if explicit:
        return explicit
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        candidate = os.path.join(rocm_path, "lib")
        if Path(candidate).exists():
            return candidate
    for candidate in ("/opt/rocm/lib", "/opt/rocm-7.2.1/lib"):
        if Path(candidate).exists():
            return candidate
    return None


# Ensure the ROCm runtime library is on the linker search path so that aiter's
# JIT compilation step can link against libamdhip64.
_rocm_lib = _resolve_rocm_lib()
if _rocm_lib:
    _cur = os.environ.get("LIBRARY_PATH", "")
    if _rocm_lib not in _cur.split(":"):
        os.environ["LIBRARY_PATH"] = f"{_rocm_lib}:{_cur}" if _cur else _rocm_lib

# Fallback: if aiter was not installed into the active venv via setup.sh,
# add the local clone to sys.path so the module is still importable.
_AITER_SRC = Path(__file__).parent.parent / "aiter_src"
if _AITER_SRC.exists() and str(_AITER_SRC) not in sys.path:
    sys.path.insert(0, str(_AITER_SRC))

from aiter.tuned_gemm import tgemm  # noqa: E402


class AiterBenchmark:
    """Profiles aiter's TunedGemm GEMM kernel for a given GEMMConfig."""

    def __init__(self, config: GEMMConfig) -> None:
        self.config = config

    def run(self, num_its: int = 10, warmup: int = 5) -> dict:
        """Profile the kernel and return timing results.

        Returns:
            {"ms": float, "tflops": float}
        """
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA/ROCm device found")
        if self.config.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"aiter only supports f16/bf16 input; got {self.config.dtype}"
            )

        cfg = self.config
        device = torch.device("cuda", torch.cuda.current_device())

        A = cfg.make_a(device)
        W = cfg.make_b(
            device
        )  # shape (n, k) — pre-transposed, matches aiter convention

        ms = profile_fn(
            lambda: tgemm.mm(A, W, otype=cfg.compute_dtype),
            num_its=num_its,
            warmup=warmup,
            print_profile=True,
        )
        return {"ms": ms, "tflops": cfg.tflops(ms)}
