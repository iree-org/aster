"""
Execute CDNA3 matmul kernel from HSACO file
===========================================

Example usage:
python ex_10_cdna3_matmul_exec.py --input test.hsaco --mcpu gfx942 \
    --num-workgroups 1 --num-wavefronts 1

Note: Default kernel name is "kernel", override with --kernel-name <name>.
"""

import argparse

from runtime import execute_kernel_from_hsaco_file
from utils import add_common_cli_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute CDNA3 matmul kernel from HSACO file"
    )

    # Add common CLI arguments (mcpu, wavefront-size, kernel-name)
    add_common_cli_args(parser)

    # Add execution-specific arguments
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input .hsaco file path"
    )
    parser.add_argument(
        "--num-workgroups",
        type=int,
        default=1,
        help="Number of workgroups (default: 1)",
    )
    parser.add_argument(
        "--num-wavefronts",
        type=int,
        default=1,
        help="Number of wavefronts per workgroup (default: 1)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of times to launch the kernel (default: 1)",
    )

    args = parser.parse_args()

    # Print configuration
    print("=== Execution Configuration ===")
    print(f"  input: {args.input}")
    print(f"  kernel_name: {args.kernel_name}")
    print(f"  mcpu: {args.mcpu}")
    print(f"  wavefront_size: {args.wavefront_size}")
    print(f"  num_workgroups: {args.num_workgroups}")
    print(f"  num_wavefronts: {args.num_wavefronts}")
    print(f"  num_iterations: {args.num_iterations}")
    print("===============================\n")

    # Execute kernel without parameters
    execute_kernel_from_hsaco_file(
        args.input,
        args.kernel_name,
        args.num_workgroups,
        args.num_wavefronts,
        args.wavefront_size,
        num_iterations=args.num_iterations,
    )

    print("\n=== Execution Complete ===")
