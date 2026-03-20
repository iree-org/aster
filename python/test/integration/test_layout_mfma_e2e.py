# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from aster import ir, utils
from aster.layout import Layout, Swizzle
from aster.dialects.kernel_builder import KernelBuilder
from aster.dialects.amdgcn import AccessKind
from aster.testing import (
    compile_mlir_module_to_asm,
    execute_kernel_and_verify,
    hsaco_file,
)

MCPU = "gfx942"

# ---------------------------------------------------------------------------
# Step 1: swizzled global load -> MFMA
# ---------------------------------------------------------------------------

# MFMA 16x16x16 f16 fragment layouts (first-mode-slowest).
# A/B: (4 groups of 16 lanes) x 8 bytes per group, 32 bytes between groups.
MFMA_AB_LAYOUT = Layout(sizes=(4, 16), strides=(8, 32))
# C/D: (4 groups of 16 lanes) x 16 bytes per group, 64 bytes between groups.
MFMA_C_LAYOUT = Layout(sizes=(4, 16), strides=(16, 64))


def _build_mfma_kernel(name, target=MCPU, isa="cdna3"):
    """Single-wave MFMA 16x16x16 f16: D = A @ B^T + C."""
    b = KernelBuilder(f"{name}_mod", name, target=target, isa=isa)
    b.add_ptr_arg(AccessKind.ReadOnly)  # A
    b.add_ptr_arg(AccessKind.ReadOnly)  # B
    b.add_ptr_arg(AccessKind.WriteOnly)  # C/D
    a_ptr, b_ptr, c_ptr = b.load_args()

    tid = b.global_thread_id()

    # Note: global load directly in mfma layout is not coalesced, this is a correctness check.
    ab_off = b.linearize_layout(tid, MFMA_AB_LAYOUT)
    c_off = b.linearize_layout(tid, MFMA_C_LAYOUT)

    a_addr = b.global_addr(a_ptr, ab_off)
    b_addr = b.global_addr(b_ptr, ab_off)
    c_addr = b.global_addr(c_ptr, c_off)

    a_frag = b.global_load_dwordx2(a_addr)
    b_frag = b.global_load_dwordx2(b_addr)
    b.wait_vmcnt(0)

    acc = b.init_agprx4(b.constant_i32(0))
    acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_frag, b_frag)

    b.global_store_dwordx4(acc, c_addr)
    b.wait_vmcnt(0)
    return b.build()


def _run_mfma_test(name):
    A = (np.random.default_rng(42).standard_normal(16 * 16) * 0.1).astype(np.float16)
    B = (np.random.default_rng(43).standard_normal(16 * 16) * 0.1).astype(np.float16)
    C = np.zeros(16 * 16, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_mfma_kernel(name)
        asm = compile_mlir_module_to_asm(module)

    path = utils.assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler does not support {MCPU}")

    with hsaco_file(path):
        if not utils.system_has_mcpu(mcpu=MCPU):
            pytest.skip(f"{MCPU} GPU not available")
        execute_kernel_and_verify(
            hsaco_path=path,
            kernel_name=name,
            input_args=[A, B],
            output_args=[C],
            mcpu=MCPU,
            wavefront_size=64,
            grid_dim=(1, 1, 1),
            block_dim=(64, 1, 1),
        )

    # MFMA 16x16x16 computes D = A @ B^T (B stored row-major [N][K]).
    # Result is transposed because MFMA C layout stores m_group in lane//16.
    ref = (A.reshape(16, 16) @ B.reshape(16, 16).T).T
    np.testing.assert_allclose(C.reshape(16, 16), ref, rtol=1e-2, atol=1e-2)


def test_mfma_16x16x16_f16():
    _run_mfma_test("mfma_16x16")


# ---------------------------------------------------------------------------
# Step 2: coalesced global load -> swizzled LDS -> MFMA fragment read -> MFMA
# ---------------------------------------------------------------------------

COALESCED_AB = Layout(sizes=64, strides=8)  # contiguous 8 bytes/lane
LDS_SWIZZLE = Swizzle(bits=2, base=4, shift=6)
LDS_B_BASE = 512  # A occupies bytes 0..511, B occupies 512..1023


def _build_mfma_lds_kernel(name, target=MCPU, isa="cdna3"):
    """Coalesced global load -> swizzled LDS write -> MFMA-layout LDS read -> MFMA."""
    b = KernelBuilder(f"{name}_mod", name, target=target, isa=isa)
    b.set_shared_memory_size(1024)  # 512 A + 512 B
    b.add_ptr_arg(AccessKind.ReadOnly)  # A
    b.add_ptr_arg(AccessKind.ReadOnly)  # B
    b.add_ptr_arg(AccessKind.WriteOnly)  # C/D
    a_ptr, b_ptr, c_ptr = b.load_args()

    tid = b.global_thread_id()

    # Stage 1: coalesced global load (contiguous 8 bytes/lane)
    coalesced_off = b.linearize_layout(tid, COALESCED_AB)
    a_addr = b.global_addr(a_ptr, coalesced_off)
    b_addr = b.global_addr(b_ptr, coalesced_off)
    a_data = b.global_load_dwordx2(a_addr)
    b_data = b.global_load_dwordx2(b_addr)
    b.wait_vmcnt(0)

    # Stage 2: swizzled LDS write (coalesced layout + swizzle for bank conflict avoidance)
    lds_write_off = b.index_to_vgpr(b.linearize_layout(tid, COALESCED_AB, LDS_SWIZZLE))
    b.ds_write_b64(a_data, lds_write_off)
    b.ds_write_b64(b_data, lds_write_off, const_offset=b.constant_i32(LDS_B_BASE))
    b.wait_lgkmcnt(0)

    # Stage 3: MFMA-layout LDS read (same swizzle)
    lds_read_off = b.index_to_vgpr(b.linearize_layout(tid, MFMA_AB_LAYOUT, LDS_SWIZZLE))
    a_frag = b.ds_read_b64(lds_read_off)
    b_frag = b.ds_read_b64(lds_read_off, const_offset=b.constant_i32(LDS_B_BASE))
    b.wait_lgkmcnt(0)

    # Stage 4: MFMA + global store at MFMA C layout
    acc = b.init_agprx4(b.constant_i32(0))
    acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_frag, b_frag)
    c_addr = b.global_addr(c_ptr, b.linearize_layout(tid, MFMA_C_LAYOUT))
    b.global_store_dwordx4(acc, c_addr)
    b.wait_vmcnt(0)
    return b.build()


def _run_mfma_lds_test(name):
    A = (np.random.default_rng(42).standard_normal(16 * 16) * 0.1).astype(np.float16)
    B = (np.random.default_rng(43).standard_normal(16 * 16) * 0.1).astype(np.float16)
    C = np.zeros(16 * 16, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_mfma_lds_kernel(name)
        asm = compile_mlir_module_to_asm(module)

    path = utils.assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler does not support {MCPU}")

    with hsaco_file(path):
        if not utils.system_has_mcpu(mcpu=MCPU):
            pytest.skip(f"{MCPU} GPU not available")
        execute_kernel_and_verify(
            hsaco_path=path,
            kernel_name=name,
            input_args=[A, B],
            output_args=[C],
            mcpu=MCPU,
            wavefront_size=64,
            grid_dim=(1, 1, 1),
            block_dim=(64, 1, 1),
        )

    ref = (A.reshape(16, 16) @ B.reshape(16, 16).T).T
    np.testing.assert_allclose(C.reshape(16, 16), ref, rtol=1e-2, atol=1e-2)


def test_mfma_16x16x16_f16_lds():
    _run_mfma_lds_test("mfma_lds")


# ---------------------------------------------------------------------------
# Step 3: multi-WG multi-wave MFMA with LDS swizzle relayout
# ---------------------------------------------------------------------------

# 3x2 waves per WG, 5x8 WGs. Each wave computes one 16x16 MFMA tile.
# Non-power-of-2 dims (3, 5) appear only in strides, not in delinearize divisors.
# All tid decompositions use power-of-2: /64 (lane), %2 and /2 (wv_x, wv_y).

N_WV_X, N_WV_Y = 2, 3
N_WG_X, N_WG_Y = 8, 5
TILE = 16
WAVES_PER_WG = N_WV_X * N_WV_Y  # 6
THREADS_PER_WG = WAVES_PER_WG * 64  # 384
N_TILES_M = N_WG_Y * N_WV_Y  # 15
N_TILES_N = N_WG_X * N_WV_X  # 16
TOTAL_M = N_TILES_M * TILE  # 240
TOTAL_N = N_TILES_N * TILE  # 256
TILE_AB_BYTES = TILE * TILE * 2  # 512 (16x16 f16)
TILE_C_BYTES = TILE * TILE * 4  # 1024 (16x16 f32)
LDS_PER_WAVE = TILE_AB_BYTES * 2  # 1024 (512 A + 512 B)
MWG_LDS_SIZE = WAVES_PER_WG * LDS_PER_WAVE  # 6144


def _build_mfma_multiwg_kernel(name, target=MCPU, isa="cdna3"):
    """Multi-WG multi-wave MFMA: coalesced load -> LDS relayout -> MFMA -> store.

    Grid: (N_WG_X, N_WG_Y, 1), Block: (THREADS_PER_WG, 1, 1).
    A is [TOTAL_M, 16] f16 row-major.  Each wave's A tile: rows [m_start:m_start+16].
    B is [TOTAL_N, 16] f16 row-major.  Each wave's B tile: rows [n_start:n_start+16].
    C is tiled: N_TILES_M * N_TILES_N tiles of 16x16 f32, each tile contiguous.
    """
    b = KernelBuilder(f"{name}_mod", name, target=target, isa=isa)
    b.set_shared_memory_size(MWG_LDS_SIZE)
    b.add_ptr_arg(AccessKind.ReadOnly)  # A
    b.add_ptr_arg(AccessKind.ReadOnly)  # B
    b.add_ptr_arg(AccessKind.WriteOnly)  # C
    a_ptr, b_ptr, c_ptr = b.load_args()

    # Decompose linear_thread_id into (wave_m, wave_n, lane)
    wv_m, wv_n, lane = b.delinearize_index(b.linear_thread_id(), (N_WV_Y, N_WV_X, 64))
    wid = b.wave_id()

    # Grid composition: tile coords from block_id + wave coords
    wm = ir.AffineExpr.get_dim(0)
    bid = ir.AffineExpr.get_symbol(0)
    tile_m = b.affine_apply(bid * N_WV_Y + wm, [wv_m], [b.block_id("y")])
    tile_n = b.affine_apply(bid * N_WV_X + wm, [wv_n], [b.block_id("x")])

    # Global byte offsets via Layout strides
    a_global = Layout(sizes=(N_TILES_M, 64), strides=(TILE_AB_BYTES, 8))
    b_global = Layout(sizes=(N_TILES_N, 64), strides=(TILE_AB_BYTES, 8))
    a_off = b.linearize_layout(
        b.linearize_index([tile_m, lane], a_global.sizes), a_global
    )
    b_off = b.linearize_layout(
        b.linearize_index([tile_n, lane], b_global.sizes), b_global
    )

    # Stage 1: coalesced global load
    a_data = b.global_load_dwordx2(b.global_addr(a_ptr, a_off))
    b_data = b.global_load_dwordx2(b.global_addr(b_ptr, b_off))
    b.wait_vmcnt(0)

    # Stage 2: swizzled LDS write
    lds_wave_layout = Layout(sizes=WAVES_PER_WG, strides=LDS_PER_WAVE)
    local, wave_base = (ir.AffineExpr.get_dim(i) for i in range(2))
    lds_write_local_v = b.linearize_layout(lane, COALESCED_AB, LDS_SWIZZLE)
    wave_base_v = b.linearize_layout(wid, lds_wave_layout)
    lds_write_off = b.affine_apply(local + wave_base, [lds_write_local_v, wave_base_v])
    lds_write_voff = b.index_to_vgpr(lds_write_off)
    b.ds_write_b64(a_data, lds_write_voff)
    b.ds_write_b64(b_data, lds_write_voff, const_offset=b.constant_i32(LDS_B_BASE))
    b.wait_lgkmcnt(0)

    # Stage 3: MFMA-layout LDS read
    lds_read_local_v = b.linearize_layout(lane, MFMA_AB_LAYOUT, LDS_SWIZZLE)
    lds_read_off = b.affine_apply(local + wave_base, [lds_read_local_v, wave_base_v])
    lds_read_voff = b.index_to_vgpr(lds_read_off)
    a_frag = b.ds_read_b64(lds_read_voff)
    b_frag = b.ds_read_b64(lds_read_voff, const_offset=b.constant_i32(LDS_B_BASE))
    b.wait_lgkmcnt(0)

    # Stage 4: MFMA
    acc = b.init_agprx4(b.constant_i32(0))
    acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_frag, b_frag)

    # Stage 5: global store
    c_tile_layout = Layout(
        sizes=(N_TILES_M, N_TILES_N), strides=(N_TILES_N * TILE_C_BYTES, TILE_C_BYTES)
    )
    tile_off, local_off = (ir.AffineExpr.get_dim(i) for i in range(2))
    c_off_v = b.linearize_layout(
        b.linearize_index([tile_m, tile_n], c_tile_layout.sizes), c_tile_layout
    )
    c_local_v = b.linearize_layout(lane, MFMA_C_LAYOUT)
    c_total = b.affine_apply(tile_off + local_off, [c_off_v, c_local_v])
    b.global_store_dwordx4(acc, b.global_addr(c_ptr, c_total))
    b.wait_vmcnt(0)
    return b.build()


def _run_mfma_multiwg_test(name):
    rng = np.random.default_rng(42)
    # A: [TOTAL_M, 16] f16, B: [TOTAL_N, 16] f16
    A = (rng.standard_normal(TOTAL_M * TILE) * 0.1).astype(np.float16)
    B = (rng.standard_normal(TOTAL_N * TILE) * 0.1).astype(np.float16)
    # C: tiled, N_TILES_M * N_TILES_N tiles of 256 f32 each
    C = np.zeros(N_TILES_M * N_TILES_N * TILE * TILE, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_mfma_multiwg_kernel(name)
        asm = compile_mlir_module_to_asm(module)

    path = utils.assemble_to_hsaco(asm, target=MCPU, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler does not support {MCPU}")

    with hsaco_file(path):
        if not utils.system_has_mcpu(mcpu=MCPU):
            pytest.skip(f"{MCPU} GPU not available")
        execute_kernel_and_verify(
            hsaco_path=path,
            kernel_name=name,
            input_args=[A, B],
            output_args=[C],
            mcpu=MCPU,
            wavefront_size=64,
            grid_dim=(N_WG_X, N_WG_Y, 1),
            block_dim=(THREADS_PER_WG, 1, 1),
        )

    A_mat = A.reshape(TOTAL_M, TILE)  # [240, 16]
    B_mat = B.reshape(TOTAL_N, TILE)  # [256, 16]

    for tm in range(N_TILES_M):
        for tn in range(N_TILES_N):
            tile_idx = tm * N_TILES_N + tn
            c_tile = C[tile_idx * 256 : (tile_idx + 1) * 256].reshape(16, 16)
            a_tile = A_mat[tm * 16 : (tm + 1) * 16, :]
            b_tile = B_mat[tn * 16 : (tn + 1) * 16, :]
            # MFMA C layout stores D^T (see step 1 verification)
            ref = (a_tile @ b_tile.T).T
            np.testing.assert_allclose(
                c_tile,
                ref,
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"tile ({tm},{tn}) mismatch",
            )


def test_mfma_16x16x16_f16_multiwg():
    _run_mfma_multiwg_test("mfma_mwg")
