"""2D memcopy benchmark using buffer_load_dwordx4 / buffer_store_dwordx4.

Copies an M x N matrix of i32 elements from src to dst in global memory.
Work is distributed across a grid of blocks, each processing a TM x TN tile.
Within a block, waves cooperatively load and store using coalesced dwordx4
buffer operations.

Two code paths:
  1. Exact -- tile sizes divide matrix dimensions, no bounds checks.
  2. OOB   -- tile sizes don't divide, lanes whose (row, col) exceed
     (m, n) get their voffset masked to 2^31 so that buffer hardware
     silently discards the access.

Parameters:
  m, n       -- matrix rows and columns (i32 elements).
  TM, TN     -- tile sizes (elements) processed by one workgroup.
  num_waves  -- waves per workgroup (block_dim = num_waves * 64).
"""

import argparse
import tempfile

import numpy as np
import pytest

from aster import ir
from aster.dialects.kernel_builder import KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.dialects._arith_enum_gen import CmpIPredicate
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.utils import system_has_mcpu

WAVE_SIZE = 64
ELEMS_PER_DWORDX4 = 4
ELEM_BYTES = 4
KERNEL_NAME = "memcopy_2d"
MCPU = "gfx942"

# Max static-loop iterations before switching to a dynamic scf.for.
_MAX_STATIC_ITERS = 16

# OOB sentinel -- any voffset >= num_records is silently discarded by
# buffer hardware (returns 0 on load, drops on store).
_OOB_VOFFSET = 1 << 31


def _ceildiv(a, b):
    return (a + b - 1) // b


def _build_memcopy_kernel(m, n, tm, tn, num_waves, mcpu=MCPU):
    """Build the IR module for a 2D tiled memcopy kernel.

    Automatically selects the exact (no bounds check) or OOB (masked
    voffset) path depending on whether tile sizes divide matrix dims.
    """
    exact = (m % tm == 0) and (n % tn == 0)
    grid_m = m // tm if exact else _ceildiv(m, tm)
    grid_n = n // tn if exact else _ceildiv(n, tn)
    num_threads = num_waves * WAVE_SIZE
    tile_elems = tm * tn
    elems_per_wave_load = WAVE_SIZE * ELEMS_PER_DWORDX4
    elems_per_block_load = num_waves * elems_per_wave_load
    assert tile_elems % elems_per_block_load == 0, (
        f"tile elements ({tile_elems}) must be divisible by elements per block load ({elems_per_block_load})"
    )
    n_iters = tile_elems // elems_per_block_load
    total_bytes = m * n * ELEM_BYTES

    b = KernelBuilder("memcopy_mod", KERNEL_NAME, target=mcpu)
    b.set_block_dims(num_threads)
    b.set_grid_dims(grid_m * grid_n)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    src_ptr, dst_ptr = b.load_args()

    num_records = b.s_mov_b32(total_bytes)
    src_rsrc = b.make_buffer_rsrc(src_ptr, num_records, b.constant_i32(0))
    dst_rsrc = b.make_buffer_rsrc(dst_ptr, num_records, b.constant_i32(0))

    wg_i, wg_j = b.delinearize_index(b.linear_block_id(), (grid_m, grid_n))
    wid = b.wave_id(wave_size=WAVE_SIZE)
    lid = b.lane_id(wave_size=WAVE_SIZE)

    d0 = ir.AffineExpr.get_dim(0)
    d1 = ir.AffineExpr.get_dim(1)
    d2 = ir.AffineExpr.get_dim(2)

    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    c_n_iters = b.constant_index(n_iters)

    use_static = n_iters <= _MAX_STATIC_ITERS
    loop_decorator = b.static_loop if use_static else b.loop

    if exact:
        # soffset: uniform byte offset of this workgroup's tile.
        tile_byte_off = b.affine_apply((d0 * grid_n + d1) * (tile_elems * ELEM_BYTES), [wg_i, wg_j])
        soffset = b.index_to_sgpr(tile_byte_off)
        lane_byte_base = b.affine_apply(d0 * (ELEMS_PER_DWORDX4 * ELEM_BYTES), [lid])

        @loop_decorator(c0, c_n_iters, c1)
        def _(it):
            wave_off = b.affine_apply(
                (d0 * num_waves + d1) * (elems_per_wave_load * ELEM_BYTES),
                [it, wid],
            )
            voff = b.affine_apply(d0 + d1, [wave_off, lane_byte_base])
            voffset = b.index_to_vgpr(voff)
            data, load_tok = b.buffer_load_dwordx4(src_rsrc, soffset, voffset)
            b.wait_deps(load_tok)
            b.buffer_store_dwordx4(data, dst_rsrc, soffset, voffset)
    else:
        from aster.dialects import arith

        soffset = b.s_mov_b32(0)

        # Precompute loop-invariant row/col bases.
        row_base = b.affine_apply(d0 * tm, [wg_i])
        col_base = b.affine_apply(d0 * tn, [wg_j])

        c_m = b.constant_index(m)

        i32_type = ir.IntegerType.get_signless(32, b._ctx)
        oob_i32 = arith.constant(i32_type, _OOB_VOFFSET, loc=b._loc, ip=b._kip)
        n_i32 = b.index_cast_i32(b.constant_index(n))

        # Per-lane column within the tile is loop-invariant.
        lane_local_col = b.affine_apply(
            (d0 * elems_per_wave_load + d1 * ELEMS_PER_DWORDX4) % tn,
            [wid, lid],
        )
        col_global = b.affine_apply(d0 + d1, [col_base, lane_local_col])
        col_i32 = b.index_cast_i32(col_global)

        col_ok = arith.cmpi(CmpIPredicate.ult, col_i32, n_i32, loc=b._loc, ip=b._kip)

        @loop_decorator(c0, c_n_iters, c1)
        def _(it):
            # Linear element index within tile for this lane.
            local_elem = b.affine_apply(
                (d0 * num_waves + d1) * elems_per_wave_load + d2 * ELEMS_PER_DWORDX4,
                [it, wid, lid],
            )
            local_row = b.affine_apply(ir.AffineExpr.get_floor_div(d0, tn), [local_elem])
            local_col = b.affine_apply(d0 % tn, [local_elem])

            row = b.affine_apply(d0 + d1, [row_base, local_row])
            col = b.affine_apply(d0 + d1, [col_base, local_col])

            # Global byte offset: (row * n + col) * ELEM_BYTES.
            voff = b.affine_apply((d0 * n + d1) * ELEM_BYTES, [row, col])

            row_i32 = b.index_cast_i32(row)
            m_i32 = b.index_cast_i32(c_m)
            voff_i32 = b.index_cast_i32(voff)

            row_ok = arith.cmpi(CmpIPredicate.ult, row_i32, m_i32, loc=b._loc, ip=b._kip)
            in_bounds = arith.andi(row_ok, col_ok, loc=b._loc, ip=b._kip)
            masked_voff = arith.select(in_bounds, voff_i32, oob_i32, loc=b._loc, ip=b._kip)
            voffset = b.index_to_vgpr(
                arith.index_cast(
                    ir.IndexType.get(b._ctx),
                    masked_voff,
                    loc=b._loc,
                    ip=b._kip,
                )
            )

            data, load_tok = b.buffer_load_dwordx4(src_rsrc, soffset, voffset)
            b.wait_deps(load_tok)
            b.buffer_store_dwordx4(data, dst_rsrc, soffset, voffset)

    b.wait_vmcnt(0)
    return b.build()


def compile_memcopy(
    m,
    n,
    tm,
    tn,
    num_waves,
    mcpu=MCPU,
    output_path=None,
    print_asm=False,
    print_ir_after_all=False,
):
    """Compile a memcopy kernel to HSACO.

    Returns (hsaco_path, asm_text).
    """
    from aster.compiler.core import PrintOptions

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_memcopy_kernel(m, n, tm, tn, num_waves, mcpu=mcpu)
        asm = compile_mlir_module_to_asm(
            module,
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=print_ir_after_all,
                print_asm=print_asm,
            ),
        )
    path = assemble_to_hsaco(
        asm,
        target=mcpu,
        wavefront_size=64,
        output_path=output_path,
    )
    assert path is not None, "assemble_to_hsaco returned None"
    return path, asm


def run_memcopy(
    m,
    n,
    tm,
    tn,
    num_waves,
    hsaco_path,
    num_iterations=1,
    src=None,
):
    """Execute a compiled memcopy HSACO.

    Returns (dst_output, times_ns).
    """
    exact = (m % tm == 0) and (n % tn == 0)
    grid_m = m // tm if exact else _ceildiv(m, tm)
    grid_n = n // tn if exact else _ceildiv(n, tn)
    num_threads = num_waves * WAVE_SIZE

    if src is None:
        np.random.seed(42)
        src = np.random.randint(0, 1000, size=m * n, dtype=np.int32)

    dst = np.zeros(m * n, dtype=np.int32)

    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=KERNEL_NAME,
        arguments=[InputArray(src), OutputArray(dst)],
        grid_dim=(grid_m * grid_n, 1, 1),
        block_dim=(num_threads, 1, 1),
        num_iterations=num_iterations,
    )
    return dst, times_ns


class TestMemcopy2D:
    """Correctness tests for the 2D memcopy kernel."""

    @pytest.mark.parametrize(
        "m,n,tm,tn,num_waves",
        [
            # Exact (tile divides matrix).
            (256, 256, 256, 256, 1),
            (512, 512, 256, 256, 4),
            (1024, 1024, 256, 256, 4),
            (1024, 512, 512, 256, 2),
            # OOB (tile does not divide matrix).
            (300, 300, 256, 256, 1),
            (500, 700, 256, 256, 4),
            (1000, 1000, 256, 256, 4),
            (100, 512, 256, 256, 1),
        ],
        ids=[
            "256x256_1w",
            "512x512_4w",
            "1024x1024_4w",
            "1024x512_2w",
            "300x300_1w_oob",
            "500x700_4w_oob",
            "1000x1000_4w_oob",
            "100x512_1w_oob",
        ],
    )
    def test_correctness(self, m, n, tm, tn, num_waves):
        if not system_has_mcpu(mcpu=MCPU):
            pytest.skip(f"{MCPU} GPU not available")

        np.random.seed(42 + m + n)
        src = np.random.randint(0, 1000, size=m * n, dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            compile_memcopy(m, n, tm, tn, num_waves, output_path=tmp.name)
            dst, _ = run_memcopy(m, n, tm, tn, num_waves, tmp.name, src=src)

        np.testing.assert_array_equal(dst, src)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D memcopy benchmark")
    parser.add_argument("--m", type=int, default=1024, help="Matrix rows")
    parser.add_argument("--n", type=int, default=1024, help="Matrix cols")
    parser.add_argument("--tm", type=int, default=256, help="Tile rows")
    parser.add_argument("--tn", type=int, default=256, help="Tile cols")
    parser.add_argument("--num-waves", type=int, default=4, help="Waves per block")
    parser.add_argument("--num-iterations", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    args = parser.parse_args()

    m, n = args.m, args.n
    tm, tn = args.tm, args.tn
    num_waves = args.num_waves
    total_bytes = m * n * ELEM_BYTES

    print(f"Memcopy {m}x{n} i32 ({total_bytes / 1e6:.1f} MB)")
    print(f"  Tile: {tm}x{tn}, waves/block: {num_waves}")

    np.random.seed(42)
    src = np.random.randint(0, 1000, size=m * n, dtype=np.int32)

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
        path, asm = compile_memcopy(
            m,
            n,
            tm,
            tn,
            num_waves,
            output_path=tmp.name,
            print_asm=args.print_asm,
            print_ir_after_all=args.print_ir_after_all,
        )
        if args.print_asm:
            print(asm)

        dst, times_ns = run_memcopy(
            m,
            n,
            tm,
            tn,
            num_waves,
            tmp.name,
            num_iterations=args.num_iterations,
            src=src,
        )

    np.testing.assert_array_equal(dst, src)
    print("  Correctness: PASS")

    for i, t_ns in enumerate(times_ns):
        # x2: one read + one write.
        bw_gbps = total_bytes * 2 / t_ns
        print(f"  Iter {i}: {t_ns / 1e3:.1f} us, {bw_gbps:.2f} GB/s")

    if len(times_ns) > 1:
        median_ns = float(np.median(times_ns))
        bw_gbps = total_bytes * 2 / median_ns
        print(f"  Median: {median_ns / 1e3:.1f} us, {bw_gbps:.2f} GB/s")
