"""Torch Inductor GEMM benchmark.

Compiles C = A @ B^T with torch.compile(backend="inductor", mode="max-
autotune") and profiles with the PyTorch profiler.
"""

import torch

from common.gemm_config import GEMMConfig
from common.profiling import profile_fn


def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T -- B is stored pre-transposed (shape N x K), transpose before matmul."""
    return torch.matmul(a, b.T)


class InductorBenchmark:
    """Compiles and profiles a torch-inductor GEMM for a given GEMMConfig."""

    def __init__(self, config: GEMMConfig) -> None:
        self.config = config
        self._compiled_fn = None
        self._device: torch.device | None = None

    def compile(self, device: torch.device) -> None:
        """Compile the GEMM with inductor max-autotune for *device*.

        Must be called before run().
        """
        self._device = device
        self._compiled_fn = torch.compile(
            _gemm,
            backend="inductor",
            mode="max-autotune",
        )
        # Trigger compilation and autotuning with a real input so that run()
        # only measures steady-state kernel execution.
        cfg = self.config
        a = cfg.make_a(device)
        b = cfg.make_b(device)
        self._compiled_fn(a, b)
        torch.cuda.synchronize()

    def run(
        self,
        num_its: int = 10,
        warmup: int = 5,
        print_profile: bool = True,
    ) -> dict:
        """Profile the compiled kernel and return timing results.

        Returns:
            {"ms": float, "tflops": float}
        """
        if self._compiled_fn is None:
            raise RuntimeError("Call compile() before run().")

        cfg = self.config
        device = self._device
        a = cfg.make_a(device)
        b = cfg.make_b(device)

        ms = profile_fn(
            lambda: self._compiled_fn(a, b),
            num_its=num_its,
            warmup=warmup,
            print_profile=print_profile,
        )
        return {"ms": ms, "tflops": cfg.tflops(ms)}
