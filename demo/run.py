#!/usr/bin/env python3

import argparse
import os
import sys

# Add project root and test/ to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))

from aster import ir, utils
from integration.test_utils import (
    compile_mlir_file_to_asm,
    execute_kernel_and_verify,
    hsaco_file,
)
from aster.pass_pipelines import EMPTY_PASS_PIPELINE

script_dir = os.path.dirname(os.path.abspath(__file__))
KERNEL_NAME = "kernel"


def main():
    """Compile and execute an AMDGCN kernel from MLIR."""
    parser = argparse.ArgumentParser(
        description="Compile and execute an AMDGCN kernel from MLIR"
    )
    parser.add_argument(
        "--mlir-file",
        type=str,
        default=os.path.join(script_dir, "add_10.mlir"),
        help="Path to MLIR file (default: demo/add_10.mlir)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of times to execute the kernel (default: 5)",
    )
    parser.add_argument(
        "--num-waves-per-cu",
        type=int,
        default=1,
        help="Number of waves per CU (default: 1)",
    )
    parser.add_argument(
        "--num-cus",
        type=int,
        default=304,
        help="Number of CUs (default: 304 for MI300X)",
    )
    args = parser.parse_args()

    mcpu = "gfx942"
    wavefront_size = 64

    # Compile MLIR to assembly
    with ir.Context() as ctx:
        asm_complete, _ = compile_mlir_file_to_asm(
            args.mlir_file,
            KERNEL_NAME,
            EMPTY_PASS_PIPELINE,
            ctx,
        )
        print(asm_complete)

    hsaco_path = utils.assemble_to_hsaco(
        asm_complete, target=mcpu, wavefront_size=wavefront_size
    )
    if hsaco_path is None:
        raise RuntimeError("Failed to assemble kernel to HSACO")
    print(f"Compilation successful. HSACO: {hsaco_path}")

    # Check if GPU is available
    if not utils.system_has_mcpu(mcpu=mcpu):
        print(
            f"Warning: GPU {mcpu} not available, stopping after cross-compilation only"
        )
        return

    # Execute kernel
    with hsaco_file(hsaco_path):
        iteration_times_ns = execute_kernel_and_verify(
            hsaco_path=hsaco_path,
            kernel_name=KERNEL_NAME,
            input_args=[],
            output_args=[],
            mcpu=mcpu,
            wavefront_size=wavefront_size,
            # TODO: ideally we'd want to have exactly 1 warp on the whole machine
            # but I don't find the right incantations on CDNA to precisely
            # profile the exact CU on which the kernel will be scheduled.
            # Note: on RDNA this seems more predictable.
            grid_dim=(args.num_cus, 1, 1),
            block_dim=(wavefront_size * args.num_waves_per_cu, 1, 1),
            verify_fn=None,
            num_iterations=args.num_iterations,
        )
        print("iteration_times_ns: ", iteration_times_ns)


if __name__ == "__main__":
    main()
