"""Minimal integration test for 1D copy kernel using dwordx4."""

import argparse
import os
import pytest
import numpy as np

from aster import ir, utils
from typing import List, Optional
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    DEFAULT_SROA_PASS_PIPELINE,
    hsaco_file,
)
from mlir_kernels.benchmarks.benchmark_utils import (
    format_throughput_stats,
    BenchmarkResult,
)
from mlir_kernels.benchmarks.benchmark_copy_1d import Copy1DConfig
from typing import Tuple


def compile_copy_1d_kernel(config: Copy1DConfig, mlir_file: str) -> Tuple[str, str]:
    """Compile a copy_1d kernel to HSACO.

    Args:
        config: Configuration for the kernel
        mlir_file: Path to the MLIR file

    Returns:
        Tuple of (hsaco_path, asm_complete) where hsaco_path is the path to the compiled HSACO file
    """
    with ir.Context() as ctx:

        def preprocess(x):
            x = x.replace(
                "{{NUM_ELEMENTS_PER_THREAD}}", str(config.num_elements_per_thread)
            )
            x = x.replace(
                "{{BLOCK_DIM_X}}", str(config.num_waves * config.wavefront_size)
            )
            x = x.replace("{{SCHED_DELAY_STORE}}", str(config.sched_delay_store))
            x = x.replace("{{GRID_DIM_X}}", str(config.num_workgroups))
            return x

        # Get library paths relative to the MLIR file (use absolute paths)
        # Load dependencies first: register-init and indexing before copies
        mlir_dir = os.path.dirname(os.path.abspath(mlir_file))
        library_dir = os.path.join(mlir_dir, "library", "common")
        library_paths = [
            os.path.abspath(os.path.join(library_dir, "register-init.mlir")),
            os.path.abspath(os.path.join(library_dir, "indexing.mlir")),
            os.path.abspath(os.path.join(library_dir, "copies.mlir")),
        ]

        asm_complete, module = compile_mlir_file_to_asm(
            mlir_file,
            config.kernel_name,
            config.pass_pipeline,
            ctx,
            preprocess=preprocess,
            library_paths=library_paths,
            print_ir_after_all=False,
        )

        # Assemble to hsaco
        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=config.mcpu, wavefront_size=config.wavefront_size
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble kernel to HSACO")

        return hsaco_path, asm_complete


def execute_copy_1d_kernel(
    config: Copy1DConfig,
    hsaco_path: str,
    num_iterations: int = 1,
) -> int:
    """Execute a compiled copy_1d kernel and return performance metrics.

    Args:
        config: Configuration for the kernel
        hsaco_path: Path to the compiled HSACO file
        num_iterations: Number of times to execute the kernel (default: 1)

    Returns:
        nanoseconds: Execution time in nanoseconds
    """
    # Prepare test data
    input_data = np.arange(config.total_num_elements_as_int32, dtype=np.int32)
    output_data = np.zeros(config.total_num_elements_as_int32, dtype=np.int32)
    timing_buffer_begin = np.zeros(1, dtype=np.int64)
    timing_buffer_end = np.zeros(1, dtype=np.int64)

    def verify_fn(input_args, output_args):
        expected = input_args[0]
        actual = output_args[0]
        if not np.array_equal(expected, actual):
            # Find first index where arrays differ
            diff_mask = expected != actual
            if np.any(diff_mask):
                first_diff_idx = np.flatnonzero(diff_mask)[0]
                assert False, (
                    f"Copy kernel failed at index {first_diff_idx}! "
                    f"Expected {expected[first_diff_idx]}, got {actual[first_diff_idx]}. "
                    f"Expected array: {expected}, Got array: {actual}"
                )
            else:
                assert False, f"Copy kernel failed! Expected {expected}, got {actual}"

    # Build padding list: [input_padding] + [output_data_padding, 0, 0]
    # (timing buffers get 0 padding)
    if config.padding_bytes is None:
        padding_bytes = [0, 0]
    elif len(config.padding_bytes) != 2:
        raise ValueError(
            f"padding_bytes must have 2 elements [input_padding, output_data_padding], "
            f"got {len(config.padding_bytes)}"
        )
    else:
        padding_bytes = config.padding_bytes

    per_buffer_padding = [padding_bytes[0]] + [padding_bytes[1], 0, 0]

    iteration_times_ns = execute_kernel_and_verify(
        hsaco_path=hsaco_path,
        kernel_name=config.kernel_name,
        input_args=[input_data],
        output_args=[output_data, timing_buffer_begin, timing_buffer_end],
        mcpu=config.mcpu,
        wavefront_size=config.wavefront_size,
        grid_dim=(config.num_workgroups, 1, 1),
        block_dim=(config.num_threads, 1, 1),
        verify_fn=verify_fn,
        padding_bytes=per_buffer_padding,
        num_iterations=num_iterations,
    )

    if not iteration_times_ns:
        raise RuntimeError("Kernel execution returned no iteration times")

    return iteration_times_ns[-1]


@pytest.mark.parametrize(
    # fmt: off
    "mlir_filename,kernel_name,num_workgroups,num_waves,num_elements_per_thread,element_size,sched_delay_store,pass_pipeline",
    [
        ("copy-1d-dwordx4.mlir", "copy_1d_dwordx4_static", 1, 1, 1, 16, 0, DEFAULT_SROA_PASS_PIPELINE),
        ("copy-1d-dwordx4.mlir", "copy_1d_dwordx4_static", 1, 1, 6, 16, 3, DEFAULT_SROA_PASS_PIPELINE),
        ("copy-1d-dwordx4.mlir", "copy_1d_dwordx4_static", 2, 2, 6, 16, 4, DEFAULT_SROA_PASS_PIPELINE),
        ("copy-1d-dwordx4.mlir", "copy_1d_dwordx4_static", 3, 3, 16, 16, 11, DEFAULT_SROA_PASS_PIPELINE),
        ("copy-1d-dwordx4.mlir", "copy_1d_dwordx4_static", 5, 7, 3, 16, 2, DEFAULT_SROA_PASS_PIPELINE),
        ("copy-1d-dwordx4.mlir", "copy_1d_dwordx4_static", 5, 7, 7, 16, 5, DEFAULT_SROA_PASS_PIPELINE),
        ("copy-1d-dwordx4.mlir", "copy_1d_dwordx4_static", 304, 10, 16, 16, 0, DEFAULT_SROA_PASS_PIPELINE),
        ("copy-1d-dwordx4.mlir", "copy_1d_dwordx4_static", 608, 10, 16, 16, 7, DEFAULT_SROA_PASS_PIPELINE),
    ],
    # fmt: on
)
@pytest.mark.parametrize("mcpu", ["gfx942"])
def test_copy_1d_dwordx4(
    mlir_filename: str,
    kernel_name: str,
    num_workgroups: int,
    num_waves: int,
    num_elements_per_thread: int,
    element_size: int,
    sched_delay_store: int,
    pass_pipeline: str,
    mcpu: str,
    wavefront_size: int = 64,
    padding_bytes: Optional[List[int]] = None,
    num_iterations: int = 1,
):
    """Test minimal 1D copy using dwordx4 (16 bytes per thread).

    Args:
        padding_bytes: List of padding bytes per buffer [input_padding, output_data_padding].
                       Timing buffers are excluded and get 0 padding.
    """

    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(test_dir, "..", mlir_filename)

    config = Copy1DConfig(
        _num_workgroups=num_workgroups,
        num_waves=num_waves,
        mlir_file=mlir_file,
        num_elements_per_thread=num_elements_per_thread,
        element_size=element_size,
        sched_delay_store=sched_delay_store,
        padding_bytes=padding_bytes,
        kernel_name=kernel_name,
        pass_pipeline=pass_pipeline,
        mcpu=mcpu,
        wavefront_size=wavefront_size,
    )

    # Compile kernel
    hsaco_path, asm_complete = compile_copy_1d_kernel(config, mlir_file)

    with hsaco_file(hsaco_path):
        # Skip execution if GPU doesn't match
        if not utils.system_has_mcpu(mcpu=mcpu):
            # Load module for printing
            with ir.Context() as ctx:

                def preprocess(x):
                    x = x.replace("{{GRID_DIM_X}}", str(num_workgroups))
                    x = x.replace("{{BLOCK_DIM_X}}", str(num_waves * wavefront_size))
                    x = x.replace(
                        "{{NUM_ELEMENTS_PER_THREAD}}", str(num_elements_per_thread)
                    )
                    x = x.replace("{{SCHED_DELAY_STORE}}", str(sched_delay_store))
                    return x

                from integration_test.test_utils import load_mlir_module_from_file

                module = load_mlir_module_from_file(
                    mlir_file, ctx, preprocess=preprocess
                )
            print(module)
            print(asm_complete)
            pytest.skip(
                f"GPU {mcpu} not available, but cross-compilation to HSACO succeeded"
            )

        # Execute kernel
        nanoseconds = execute_copy_1d_kernel(config, hsaco_path, num_iterations)
        result = BenchmarkResult(
            config=config,
            iteration_times_ns=[nanoseconds],
        )
        print(format_throughput_stats(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 1D copy kernel")
    parser.add_argument(
        "--num-waves",
        type=int,
        default=1,
        help="Number of waves (default: 1)",
    )
    parser.add_argument(
        "--num-workgroups",
        type=int,
        default=1,
        help="Number of workgroups (default: 1)",
    )
    parser.add_argument(
        "--num-elements-per-thread",
        type=int,
        default=16,
        help="Number of elements per thread (default: 16)",
    )
    parser.add_argument(
        "--element-size",
        type=int,
        default=16,
        help="Element size in bytes (default: 16)",
    )
    parser.add_argument(
        "--sched-delay-store",
        type=int,
        default=3,
        help="Schedule delay for store (default: 3)",
    )
    parser.add_argument(
        "--mlir-filename",
        type=str,
        default="copy-1d-dwordx4.mlir",
        help="MLIR filename (default: copy-1d-dwordx4.mlir)",
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default="copy_1d_dwordx4_static",
        help="Kernel name (default: copy_1d_dwordx4_static)",
    )
    parser.add_argument(
        "--mcpu",
        type=str,
        default="gfx942",
        help="Target GPU architecture (default: gfx942)",
    )
    parser.add_argument(
        "--wavefront-size",
        type=int,
        default=64,
        help="Wavefront size (default: 64)",
    )
    parser.add_argument(
        "--padding-bytes",
        type=int,
        nargs="+",
        default=[0, 0],
        help="Padding bytes per buffer [input_padding, output_data_padding]. "
        "Timing buffers are excluded. (default: 0 0)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of times to execute the kernel (default: 5)",
    )

    args = parser.parse_args()

    # Validate padding_bytes argument
    if len(args.padding_bytes) != 2:
        parser.error(
            f"--padding-bytes requires exactly 2 values [input_padding, output_data_padding], "
            f"got {len(args.padding_bytes)}"
        )

    test_copy_1d_dwordx4(
        mlir_filename=args.mlir_filename,
        kernel_name=args.kernel_name,
        num_elements_per_thread=args.num_elements_per_thread,
        num_workgroups=args.num_workgroups,
        num_waves=args.num_waves,
        element_size=args.element_size,
        sched_delay_store=args.sched_delay_store,
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=args.mcpu,
        wavefront_size=args.wavefront_size,
        padding_bytes=args.padding_bytes,
        num_iterations=args.num_iterations,
    )
