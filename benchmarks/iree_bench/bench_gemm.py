"""IREE GEMM benchmark.

Compiles C = A @ B^T with iree-turbine and profiles with the PyTorch profiler. Adapted
from ~/aster/gemm-bench/bench_gemm.py.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.nn as nn
import iree.turbine.aot as aot

from iree.compiler import compile_file, OutputFormat
from iree.runtime import VmModule
from iree.turbine.runtime import Launchable, Device
from iree.turbine.runtime.device import get_device_from_torch

from common.gemm_config import GEMMConfig
from common.profiling import profile_fn

DRIVER_TO_BACKEND: dict[str, str] = {
    "hip": "rocm",
    "cuda": "cuda",
    "local-task": "llvm-cpu",
    "local-sync": "llvm-cpu",
}


# ---------------------------------------------------------------------------
# GEMM nn.Module
# ---------------------------------------------------------------------------


class _GEMMModule(nn.Module):
    """Pure GEMM: C = A @ B with out=c.

    A is (m, k) in compute_dtype, B is (k, n) in compute_dtype (already
    transposed by the caller), and C is the (m, n) output buffer.
    """

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(a, b, out=c)


# ---------------------------------------------------------------------------
# IREE run wrapper
# ---------------------------------------------------------------------------


class _IREERun:
    """Calls an IREE-compiled VMFB via iree-turbine's Launchable."""

    def __init__(self, vmfb_bytes: bytes, arguments: list[Any]) -> None:
        self.arguments = arguments

        def _get_vmfb(device: Device) -> VmModule:
            return VmModule.copy_buffer(device.vm_instance, vmfb_bytes)

        self.kernel = Launchable.from_vm_module(_get_vmfb, entry_point="main")

    def __call__(self) -> Any:
        return self.kernel(*self.arguments)


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------


def _compile(
    device: torch.device,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    print_mlir: bool = False,
    print_asm: bool = False,
) -> bytes:
    """Export *module* through iree-turbine and compile to a VMFB binary."""
    exported: aot.ExportOutput = aot.export(module, args=args)

    if print_mlir:
        print("\n" + "=" * 70, flush=True)
        print("MLIR — Torch IR (before IREE compilation):", flush=True)
        print("=" * 70, flush=True)
        print(exported.mlir_module, flush=True)

    iree_device = get_device_from_torch(device)
    target_backend = DRIVER_TO_BACKEND[iree_device.driver_id]
    ireecc_args = list(iree_device.compile_target_flags) + [
        "--iree-opt-level=O3",
        "--iree-opt-strip-assertions=true",
    ]

    with TemporaryDirectory() as tmp:
        mlirbc_path = Path(tmp) / "exported.mlirbc"
        exported.save_mlir(str(mlirbc_path))

        dump_dir = Path(tmp) / "asm_dump"
        if print_asm:
            dump_dir.mkdir()
            ireecc_args = ireecc_args + [
                f"--iree-hal-dump-executable-intermediates-to={dump_dir}"
            ]

        vmfb_bytes = compile_file(
            str(mlirbc_path),
            target_backends=[target_backend],
            extra_args=ireecc_args,
            output_format=OutputFormat.FLATBUFFER_BINARY,
            optimize=True,
            strip_source_map=True,
            strip_debug_ops=True,
            output_mlir_debuginfo=False,
        )

        if print_asm:
            asm_files = sorted(dump_dir.glob("*.rocmasm"))
            if not asm_files:
                asm_files = sorted(dump_dir.glob("*.s"))
            for asm_file in asm_files:
                print("\n" + "=" * 70, flush=True)
                print(f"AMDGCN ASM — {asm_file.name}:", flush=True)
                print("=" * 70, flush=True)
                print(asm_file.read_text(), flush=True)

    return vmfb_bytes


# ---------------------------------------------------------------------------
# Public benchmark class
# ---------------------------------------------------------------------------


class IREEBenchmark:
    """Compiles and profiles an IREE GEMM for a given GEMMConfig."""

    def __init__(self, config: GEMMConfig) -> None:
        self.config = config
        self._vmfb_bytes: bytes | None = None

    def compile(
        self,
        device: torch.device,
        print_mlir: bool = False,
        print_asm: bool = False,
    ) -> None:
        """Compile the GEMM kernel for *device*.

        Must be called before run().
        """
        cfg = self.config
        a = cfg.make_a(device)
        # B is stored (n, k) pre-transposed; transpose to (k, n) for matmul.
        b = cfg.make_b(device).t().contiguous()
        c = cfg.make_c(device)

        module = _GEMMModule()
        self._vmfb_bytes = _compile(
            device,
            module,
            (a, b, c),
            print_mlir=print_mlir,
            print_asm=print_asm,
        )
        self._device = device

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
        if self._vmfb_bytes is None:
            raise RuntimeError("Call compile() before run().")

        cfg = self.config
        device = self._device

        a = cfg.make_a(device)
        b = cfg.make_b(device).t().contiguous()
        c = cfg.make_c(device)

        iree_run = _IREERun(self._vmfb_bytes, [a, b, c])

        ms = profile_fn(
            iree_run,
            num_its=num_its,
            warmup=warmup,
            print_profile=print_profile,
        )
        return {"ms": ms, "tflops": cfg.tflops(ms)}
