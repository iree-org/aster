"""Triton GEMM benchmark for AMD GPU (ROCm 7+).

Adapted from ~/aster/triton/gemm_bench.py.

Kernel optimisations:
  - No boundary masks: M, N, K and block dims assumed divisible by 32.
  - Exact integer division in the K-loop communicates to the compiler that
    there is no remainder, enabling cleaner loop unrolling.
  - @triton.autotune sweeps tile shapes, pipeline depth, and AMD knobs:
      matrix_instr_nonkdim — 16×16 or 32×32 MFMA instruction
      waves_per_eu         — occupancy hint for the wave scheduler
      kpack                — LDS K-packing factor (reduces bank conflicts)
  - num_stages > 1 enables Triton's software pipeline that overlaps HBM loads
    of the next K-block with MFMA computation on the current one.
"""

import sys

import torch
import triton
import triton.knobs as _knobs
import triton.language as tl

from common.gemm_config import GEMMConfig
from common.profiling import profile_fn

# Disable DWARF line-info globally: skips the add_di_scope LLVM pass
# (amd/compiler.py:332) so every compiled binary is debug-section-free.
_knobs.compilation.disable_line_info = True


# ---------------------------------------------------------------------------
# Autotune config builder for AMD MI300X (gfx942)
# ---------------------------------------------------------------------------


def _build_configs() -> list:
    block_mn = [
        (128, 128),
        (128, 256),
        (256, 128),
        (256, 256),
    ]
    block_k = [64, 128]
    group_m = [4, 8]
    num_stages = [1, 2]
    waves_per_eu = [0, 2]
    matrix_instr_nonkdim = [16]
    kpack = [1, 2]

    def warps_for(bm: int, bn: int) -> list:
        elems = bm * bn
        if elems <= 128 * 128:
            return [4]
        if elems <= 256 * 128:
            return [4, 8]
        return [8]

    configs = []
    for bm, bn in block_mn:
        for bk in block_k:
            for gm in group_m:
                for stages in num_stages:
                    for nw in warps_for(bm, bn):
                        for waves in waves_per_eu:
                            for instr in matrix_instr_nonkdim:
                                for kp in kpack:
                                    configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M": bm,
                                                "BLOCK_N": bn,
                                                "BLOCK_K": bk,
                                                "GROUP_M": gm,
                                                "waves_per_eu": waves,
                                                "matrix_instr_nonkdim": instr,
                                                "kpack": kp,
                                            },
                                            num_warps=nw,
                                            num_stages=stages,
                                        )
                                    )
    return configs


_CONFIGS = _build_configs()

# Set via TritonBenchmark.max_configs to randomly subsample the config space.
_MAX_CONFIGS: int | None = None


def _prune_configs(configs, named_args, **_kwargs):
    """Drop configs whose tile dims exceed the matrix dims."""
    import random

    M = min(named_args["M"], 8192)
    N = min(named_args["N"], 8192)
    K = min(named_args["K"], 8192)
    keep = [
        c
        for c in configs
        if c.kwargs["BLOCK_M"] <= M
        and c.kwargs["BLOCK_N"] <= N
        and c.kwargs["BLOCK_K"] <= K
    ]
    keep = keep or configs[:1]
    if _MAX_CONFIGS is not None and len(keep) > _MAX_CONFIGS:
        keep = random.sample(keep, _MAX_CONFIGS)
    return keep


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit(debug=False)
def _gemm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C = A @ B — B is expected pre-transposed (shape N×K, row-major).

    No boundary guards: M, N, K are divisible by the block dims.
    """
    tl.assume(M % BLOCK_M == 0)
    tl.assume(N % BLOCK_N == 0)
    tl.assume(K % BLOCK_K == 0)
    tl.assume(M <= 8192)
    tl.assume(N <= 8192)
    tl.assume(K <= 8192)

    pid = tl.program_id(0)
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, acc)


def _triton_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T — B must be pre-transposed: shape (N, K), contiguous."""
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    N, _ = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (M // meta["BLOCK_M"] * (N // meta["BLOCK_N"]),)  # noqa: E731
    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        C.stride(0),
        C.stride(1),
    )
    return C


# ---------------------------------------------------------------------------
# Assembly printing helper
# ---------------------------------------------------------------------------


def print_asm(M: int, N: int, K: int) -> None:
    """Autotune and dump the AMDGCN ISA of the winning config."""
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16)

    _triton_gemm(A, B)
    best = _gemm_kernel.best_config

    _gemm_kernel.fn.device_caches.clear()
    _triton_gemm(A, B)

    printed = False
    for cache_tuple in _gemm_kernel.fn.device_caches.values():
        for compiled in cache_tuple[0].values():
            asm = compiled.asm
            if "amdgcn" in asm:
                print(f"\n{'='*70}")
                print(f"AMDGCN ISA  M={M} N={N} K={K}")
                print(f"Config: {best}")
                print(f"{'='*70}")
                print(asm["amdgcn"])
                printed = True
            elif "ptx" in asm:
                print(asm["ptx"])
                printed = True
    if not printed:
        print("[WARN] No ISA found in kernel cache.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public benchmark class
# ---------------------------------------------------------------------------


class TritonBenchmark:
    """Profiles a Triton GEMM kernel for a given GEMMConfig."""

    def __init__(self, config: GEMMConfig, max_configs: int | None = None) -> None:
        self.config = config
        self.max_configs = max_configs

    def run(
        self,
        warmup: int = 5,
        iters: int = 10,
        do_print_asm: bool = False,
    ) -> dict:
        """Profile the kernel and return timing results.

        Returns:
            {"ms": float, "tflops": float, "best_config": str}
        """
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA/ROCm device found.")

        global _MAX_CONFIGS
        _MAX_CONFIGS = self.max_configs

        cfg = self.config
        device = torch.device("cuda", torch.cuda.current_device())

        A = cfg.make_a(device)
        B = cfg.make_b(device)  # shape (n, k) — pre-transposed

        if do_print_asm:
            print_asm(cfg.m, cfg.n, cfg.k)

        ms = profile_fn(
            lambda: _triton_gemm(A, B),
            num_its=iters,
            warmup=warmup,
            print_profile=True,
        )
        best = getattr(_gemm_kernel, "best_config", None)
        return {
            "ms": ms,
            "tflops": cfg.tflops(ms),
            "best_config": str(best),
        }
