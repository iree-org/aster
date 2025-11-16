"""
Saturating the MATRIX unit
==========================

Example usage:

python ex_10_cdna3_matmul_v3.py --output test.hsaco --mcpu gfx942 \
    --m-regs 1 --n-regs 1 --k-regs 1 --iteration-step-size 1 --prefetch-amount 1 \
    --num-vgprs=16  --dump-asm

python ex_10_cdna3_matmul_v3.py --output test.hsaco --mcpu gfx942 \
    --m-regs 2 --n-regs 2 --k-regs 8 --iteration-step-size 4 --prefetch-amount 8 \
    --num-vgprs=144 --num-agprs=16 --dump-ir --dump-asm

python ex_10_cdna3_matmul_v3.py --output test.hsaco --mcpu gfx942 \
    --m-regs 4 --n-regs 4 --k-regs 4 --iteration-step-size 4 --prefetch-amount 8 \
    --num-vgprs=120 --num-agprs=64

python ex_10_cdna3_matmul_v3.py --output test.hsaco --mcpu gfx942 \
    --m-regs 4 --n-regs 4 --k-regs 4 --iteration-step-size 4 --prefetch-amount 8 \
    --num-vgprs=120 --num-agprs=64 --use-relocatable-registers

Note: v_mfma_f32_16x16x16_f16 requires 2 VGPRs for A and B operands, 4 AGPRs for C accumulator.
      Default kernel name is "kernel", override with --kernel-name <name>.
"""

from math import prod

from aster import ir
from aster.dialects import api
from aster.dialects.api import s_waitcnt
from kernel_utils import (
    ARegRangeArray,
    VRegRangeArray,
    allocate_indexing_vgprs,
    delinearize_with_permutation,
    get_strides,
    linearize,
    setup_indexing_registers,
    setup_matmul_arrays,
)


def _execute_matmul_v3_with_prefetch(
    ctx: "ir.Context",  # type: ignore
    A: VRegRangeArray,
    B: VRegRangeArray,
    C: VRegRangeArray | ARegRangeArray,
    indexing_vgprs: list,
    iteration_step_size: int = 4,
    prefetch_amount: int = 8,
    permutation: tuple[int, int, int] = (0, 1, 2),
) -> None:
    # Extract dimensions from array shapes
    m_regs, k_regs = A.shape
    assert B.shape == (k_regs, C.shape[1]), "B shape must match (k_regs, n_regs)"
    n_regs = C.shape[1]

    # TODO: option to reorder traversal order to get multiple C's in flight.
    shape = (m_regs, n_regs, k_regs)
    strides = get_strides(shape)
    num_iters = linearize(tuple(s - 1 for s in shape), strides) + 1
    assert num_iters == prod(shape)

    # Sweetspots on RDNA4 4x4x4 on 196 vgprs:
    #   - 4/2 for 1  warp (18.7 cy / wmma)
    #   - 3/1 for 2 warps (12.5 cy / wmma !)
    #   - 2/1 for 3 warps (14.6 cy / wmma !)
    #   - 4/1 for 4 warps (17.2 cy / wmma)
    ub = num_iters + prefetch_amount

    num_in_flight = [0] * (ub)

    # Preinitialize C when zero_init is True
    if C.zero_init:
        for iter in range(ub - prefetch_amount):
            load_idx = iter
            m_load, n_load, k_load = delinearize_with_permutation(
                load_idx, shape, permutation
            )
            # if c is already initialized, this does not emit a mov.
            c, inc_c = C.get((m_load, n_load))

    for slice_b in range(0, ub, iteration_step_size):
        slice_e = min(slice_b + iteration_step_size, ub)

        #########################################################
        # Prefetch loads with reuse.
        #########################################################
        for load_idx in range(slice_b, slice_e, 1):
            if load_idx < 0 or load_idx >= ub - prefetch_amount:
                continue
            m_load, n_load, k_load = delinearize_with_permutation(
                load_idx, shape, permutation
            )

            if m_load < m_regs and n_load < n_regs and k_load < k_regs:
                a, inc_a = A.get((m_load, k_load))
                b, inc_b = B.get((k_load, n_load))
                c, inc_c = C.get((m_load, n_load))
                num_in_flight[load_idx] = inc_c + inc_a + inc_b

        assert (
            prefetch_amount >= 0
        ), f"prefetch_amount must be >= 0 got {prefetch_amount}"

        #########################################################
        # Compute.
        #########################################################
        compute_idx_b = max(slice_b - prefetch_amount, 0)  # incl.
        compute_idx_e = min(slice_e - prefetch_amount, ub)  # excl.
        assert (
            compute_idx_e <= ub - prefetch_amount
        ), f"compute_idx_e is out of bounds {compute_idx_e} > {ub - prefetch_amount}"
        if compute_idx_e > 0:  # excl.
            num_loads_so_far = sum(num_in_flight)
            num_that_must_be_finished = sum(num_in_flight[0 : compute_idx_e + 1])
            at_most_so_many_in_flight = num_loads_so_far - num_that_must_be_finished
            if at_most_so_many_in_flight >= 0:
                # Use lgkmcnt for LDS operations (ds_read) and vmcnt for global loads if any
                # For now, we primarily use LDS so we use lgkmcnt
                s_waitcnt(lgkmcnt=at_most_so_many_in_flight)

        for compute_idx in range(compute_idx_b, compute_idx_e, 1):
            if compute_idx >= 0:
                m, n, k = delinearize_with_permutation(compute_idx, shape, permutation)
                a, inc_a = A.get((m, k))
                b, inc_b = B.get((k, n))
                c, inc_c = C.get((m, n))
                assert (
                    inc_c == 0 and inc_a == 0 and inc_b == 0
                ), f"need to flush loads at m, n, k = {m}, {n}, {k}"
                # Use acc_cd=True when C uses AGPRs
                C[m, n] = api.v_mfma_f32_16x16x16_f16(c, a, b, c)

        #########################################################
        # Write back when done.
        #########################################################
        for compute_idx in range(compute_idx_b, compute_idx_e, 1):
            if compute_idx >= 0:
                m, n, k = delinearize_with_permutation(compute_idx, shape, permutation)
                a, inc_a = A.get((m, k))
                b, inc_b = B.get((k, n))
                c, inc_c = C.get((m, n))
                assert (
                    inc_c == 0 and inc_a == 0 and inc_b == 0
                ), f"need to flush loads at m, n, k = {m}, {n}, {k}"
                if k == k_regs - 1:
                    assert (
                        indexing_vgprs is not None
                    ), "indexing_vgprs is required for storing"
                    assert (
                        len(indexing_vgprs) > 0
                    ), "indexing_vgprs must contain at least one VGPR"
                    # TODO: Implement global store once global_addr is set up in setup_matmul_arrays
                    # For now, stores are skipped as global addresses need to be loaded from kernel arguments
                    # vaddr_offset = indexing_vgprs[0]
                    # linear_idx = linearize((m, n), get_strides((m_regs, n_regs)))
                    # offset = linear_idx * C.rs * 4
                    # if C.global_addr is not None:
                    #     addr_range = make_register_range(list(C.global_addr))
                    #     global_store_dwordx4(c, addr_range, offset=offset)

    #########################################################
    # Wait for all stores to complete.
    #########################################################
    s_waitcnt(vmcnt=0)


def _inject_matmul_v3_on_demand_loads_with_prefetch(
    ctx: "ir.Context",  # type: ignore
    sgprs: list,
    vgprs: list,
    agprs: list,
    num_iterations: int,
    *,
    operand_register_size: int = 2,
    accum_register_size: int = 4,
    m_regs: int = 4,
    n_regs: int = 4,
    k_regs: int = 4,
    iteration_step_size: int = 4,
    prefetch_amount: int = 8,
    permutation: tuple[int, int, int] = (0, 1, 2),
) -> None:
    lds_sizes = [8192, 8192, 8192]

    unused_sgprs, unused_vgprs, unused_agprs, A, B, C = setup_matmul_arrays(
        ctx,
        sgprs,
        vgprs,
        operand_register_size,
        accum_register_size,
        m_regs,
        n_regs,
        k_regs,
        lds_sizes,
        agprs=agprs,
    )
    sgprs, vgprs = unused_sgprs, unused_vgprs

    # Allocate indexing registers
    num_indexing_regs = setup_indexing_registers()
    indexing_vgprs, unused_vgprs = allocate_indexing_vgprs(
        ctx, unused_vgprs, num_indexing_regs
    )
    vgprs = unused_vgprs

    _execute_matmul_v3_with_prefetch(
        ctx,
        A,
        B,
        C,
        indexing_vgprs,
        iteration_step_size,
        prefetch_amount,
        permutation,
    )


if __name__ == "__main__":
    import argparse
    import sys
    from ex_10_cdna3_matmul_cli import run_matmul_cli

    # Create parser only for v3-specific arguments (not matmul args)
    parser = argparse.ArgumentParser(
        description="Matrix multiplication kernel v3 with prefetching"
    )
    parser.add_argument(
        "--iteration-step-size",
        type=int,
        default=4,
        help="Iteration step size for prefetching (default: 4)",
    )
    parser.add_argument(
        "--prefetch-amount",
        type=int,
        default=8,
        help="Prefetch amount for latency hiding (default: 8)",
    )
    parser.add_argument(
        "--permutation",
        type=str,
        default="0,2,1",
        help="Permutation of (m,k,n) dimensions for iteration order, e.g., '0,2,1' for (m,n,k) (default: 0,2,1)",
    )

    # Parse v3-specific args only (matmul args will be parsed by run_matmul_cli)
    args, remaining_args = parser.parse_known_args()

    # Parse permutation string into tuple
    perm_values = [int(x) for x in args.permutation.split(",")]
    assert len(perm_values) == 3, "Permutation must have 3 elements"
    assert set(perm_values) == {
        0,
        1,
        2,
    }, "Permutation must be a permutation of (0, 1, 2)"
    permutation: tuple[int, int, int] = (perm_values[0], perm_values[1], perm_values[2])

    # Print parsed v3-specific arguments
    print("=== Matmul V3 Configuration ===")
    print(f"  iteration_step_size: {args.iteration_step_size}")
    print(f"  prefetch_amount: {args.prefetch_amount}")
    print(f"  permutation: {permutation}")
    print("================================\n")

    # Save original argv and restore args for run_matmul_cli
    # remaining_args still contains matmul args and utils args
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_args

    # Create wrapped inject function that includes v3-specific parameters
    def inject_fn(ctx, sgprs, vgprs, agprs, num_iterations, **kwargs):
        return _inject_matmul_v3_on_demand_loads_with_prefetch(
            ctx,
            sgprs,
            vgprs,
            agprs,
            num_iterations,
            **kwargs,
            iteration_step_size=args.iteration_step_size,
            prefetch_amount=args.prefetch_amount,
            permutation=permutation,
        )

    try:
        run_matmul_cli(inject_fn, add_args=True)
    finally:
        sys.argv = original_argv
