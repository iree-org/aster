"""GEMMConfig: problem-size descriptor and matrix factory for GEMM benchmarks."""

from dataclasses import dataclass, field

import torch

DTYPE_NAMES: dict[str, torch.dtype] = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

# Mapping from torch dtype to rocBLAS/hipBLAS type string used in CLI commands.
DTYPE_TO_ROCBLAS: dict[torch.dtype, str] = {
    torch.float16: "f16_r",
    torch.bfloat16: "bf16_r",
    torch.float32: "f32_r",
}


@dataclass
class GEMMConfig:
    """Describes a single GEMM problem: C = A @ B^T (when transpose_b=True).

    Matrix shapes:
      A: (m, k)  — dtype
      B: (n, k)  — dtype, stored pre-transposed when transpose_b=True
      C: (m, n)  — compute_dtype
    """

    m: int
    n: int
    k: int
    dtype: torch.dtype = torch.float16
    compute_dtype: torch.dtype = torch.float32
    # When True, B is stored as (n, k) so that the GEMM is A @ B.T with
    # B originally being (n, k). This matches the convention used in the
    # existing reference benchmarks.
    transpose_b: bool = True
    seed: int = 42

    def make_a(self, device: torch.device) -> torch.Tensor:
        """Return a random (m, k) tensor with dtype=self.dtype."""
        torch.manual_seed(self.seed)
        return torch.randn(self.m, self.k, dtype=self.dtype, device=device)

    def make_b(self, device: torch.device) -> torch.Tensor:
        """Return a random B tensor.

        Shape is (n, k) when transpose_b=True (pre-transposed), or (k, n) otherwise.
        """
        torch.manual_seed(self.seed + 1)
        if self.transpose_b:
            return torch.randn(self.n, self.k, dtype=self.dtype, device=device)
        return torch.randn(self.k, self.n, dtype=self.dtype, device=device)

    def make_c(self, device: torch.device) -> torch.Tensor:
        """Return a zero (m, n) output tensor with dtype=self.compute_dtype."""
        return torch.zeros(self.m, self.n, dtype=self.compute_dtype, device=device)

    def flops(self) -> int:
        """Return the number of floating-point operations (multiply-add = 2)."""
        return 2 * self.m * self.n * self.k

    def tflops(self, ms: float) -> float:
        """Convert a kernel time in milliseconds to TFLOP/s."""
        if ms <= 0:
            return float("nan")
        return self.flops() / (ms * 1e9)

    def dtype_name(self) -> str:
        """Return the short string name for self.dtype (e.g. 'f16')."""
        for name, dtype in DTYPE_NAMES.items():
            if dtype == self.dtype:
                return name
        return str(self.dtype)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of this config."""
        return {
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "dtype": self.dtype_name(),
            "compute_dtype": "f32",
            "transpose_b": self.transpose_b,
            "seed": self.seed,
        }
