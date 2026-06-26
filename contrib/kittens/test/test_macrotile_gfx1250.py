import os
import sys
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest

from aster import ir
from aster.dialects.kernel_builder_with_layouts import (
    ds_load_b128,
    KernelBuilderWithLayouts as KernelBuilder,
)
from aster.dialects.amdgcn import AccessKind, BarrierScope
from aster.dialects.kernel_builder import TdmPadMode
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu
from aster.pass_pipelines import make_default_pass_pipeline, PipelineConfig

from aster.layout import Layout, Symbol, Tensor, tile

MCPU = "gfx1250"
KERNEL_NAME = "gemm_macrotile"
WAVE_SIZE = 32

STAGE_LOAD = 0
STAGE_WRITE = 1
STAGE_READ = 2
STAGE_COMPUTE = 3
N_STAGES = STAGE_COMPUTE + 1


class WmmaIntrinsicKind(Enum):
    """Identifies a WMMA matrix instruction; the value is its asm mnemonic."""

    V_WMMA_F32_16x16x32_F16 = "v_wmma_f32_16x16x32_f16"


# TODO: this is gfx1250 WMMA specific atm, generalize later.
@dataclass(frozen=True)
class WmmaIntrinsic:
    """A WMMA matrix instruction: its kind/mnemonic, M/N/K tile, and the access
    maps its fragments follow in registers (centralized so callers never
    hand-roll the magic strides)."""

    kind: WmmaIntrinsicKind
    m: int
    n: int
    k: int

    @property
    def mnemonic(self) -> str:
        return self.kind.value

    def output_tile_layout(self, row_byte_stride: int, elem_bytes: int = 4) -> Layout:
        """Row-major C store layout for one accumulator output tile."""
        half = self.m // 2
        return Layout((2, self.n, half), (half * row_byte_stride, elem_bytes, row_byte_stride))

    def operand_read_layout(
        self, row_byte_stride: int, elem_bytes: int = 2, wave_size: int = 32
    ) -> tuple[Layout, Layout]:
        """LDS->register read (thread_layout, value_layout) for row-major."""

        # gfx1250 v3 WMMA packs k as kTileRepeat disjoint blocks of kWidth=8
        kwidth = 8
        kdepth = wave_size // self.m
        krepeat = self.k // (kdepth * kwidth)
        thread_layout = Layout((kdepth, self.m), (kwidth * elem_bytes, row_byte_stride))
        value_layout = Layout(krepeat, kdepth * kwidth * elem_bytes)
        return thread_layout, value_layout


WMMA_INTRINSICS = {
    WmmaIntrinsicKind.V_WMMA_F32_16x16x32_F16: WmmaIntrinsic(WmmaIntrinsicKind.V_WMMA_F32_16x16x32_F16, 16, 16, 32),
}

WMMA_INTRINSIC = WMMA_INTRINSICS[WmmaIntrinsicKind.V_WMMA_F32_16x16x32_F16]
WMMA_MNEMONIC = WMMA_INTRINSIC.mnemonic

# Tile hierarchy, smallest to largest: one WG computes a MACRO_TILE out of SUB
# WMMA subtiles (exact division); CLUSTER WGs form the per-cluster WG_TILE; a
# CLUSTER_GRID of clusters tiles the full M x N (M != N permitted).
MACRO_TILE_M, MACRO_TILE_N, MACRO_TILE_K = 64, 64, 32  # per-WG output tile
CLUSTER_M, CLUSTER_N, CLUSTER_K = 4, 4, 1  # WGs per cluster
WG_TILE_M, WG_TILE_N, WG_TILE_K = (
    MACRO_TILE_M * CLUSTER_M,
    MACRO_TILE_N * CLUSTER_N,
    MACRO_TILE_K * CLUSTER_K,
)  # per-cluster tile: 256, 256, 32
WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K = WMMA_INTRINSIC.m, WMMA_INTRINSIC.n, WMMA_INTRINSIC.k
SUB_M, SUB_N, SUB_K = (
    MACRO_TILE_M // WMMA_TILE_M,
    MACRO_TILE_N // WMMA_TILE_N,
    MACRO_TILE_K // WMMA_TILE_K,
)  # WMMA subtiles per WG: 4, 4, 1
assert (SUB_M * WMMA_TILE_M, SUB_N * WMMA_TILE_N, SUB_K * WMMA_TILE_K) == (
    MACRO_TILE_M,
    MACRO_TILE_N,
    MACRO_TILE_K,
), "MACRO_TILE must be an exact multiple of WMMA_TILE"

CLUSTER_GRID_M, CLUSTER_GRID_N = 32, 32  # clusters per dim
WG_GRID_M, WG_GRID_N = CLUSTER_GRID_M * CLUSTER_M, CLUSTER_GRID_N * CLUSTER_N  # 128, 128
NUM_WGS = WG_GRID_M * WG_GRID_N  # 16384
ROW_BYTES = MACRO_TILE_K * 2  # bytes per per-WG tile row (MACRO_TILE_K f16)
PAD_BYTES = 16  # LDS pad per row to break bank conflicts (TDM has no write-side swizzle)
M, N, K = WG_TILE_M * CLUSTER_GRID_M, WG_TILE_N * CLUSTER_GRID_N, 4096  # 8192, 8192, 4096

# WMMA subtile axes for the layout-driven LDS read (one axis per per-WG LDS
# strip: SUB_M A subtiles along M, SUB_N B subtiles along N).
m_sub = Symbol("m")
n_sub = Symbol("n")
# Global-tensor tiling axes (cluster hierarchy + K iteration), sliced per WG.
cm_ax = Symbol("cm")  # within-cluster WG index in M
cgm_ax = Symbol("cgm")  # cluster-grid index in M
cn_ax = Symbol("cn")  # within-cluster WG index in N
cgn_ax = Symbol("cgn")  # cluster-grid index in N
global_k = Symbol("global_k")  # K tile iteration


def _build_macrotile(M, N, K, *, cluster_multicast=False, pad_mode=None):
    assert K % MACRO_TILE_K == 0, f"K={K} must be divisible by {MACRO_TILE_K}"
    k_tiles = K // MACRO_TILE_K
    assert k_tiles >= N_STAGES, f"k_tiles={k_tiles} < N_STAGES={N_STAGES}"

    stride_a = K * 2  # f16 bytes per A row
    stride_b = K * 2  # f16 bytes per B row
    stride_c = N * 4  # f32 bytes per C row

    # Padded LDS row stride (TDM has no write-side swizzle): pad_mode in
    # {"padding", "iterate"} inserts PAD_BYTES per row; None keeps the dense layout.
    pad_bytes = PAD_BYTES if pad_mode else 0
    padded_row = ROW_BYTES + pad_bytes
    frag_bytes = WMMA_TILE_M * padded_row  # LDS span of one WMMA subtile (WMMA_TILE_M rows)
    lds_total_a = MACRO_TILE_M * padded_row  # per-WG A strip
    lds_total_b = MACRO_TILE_N * padded_row  # per-WG B strip
    tdm_pad = dict(lds_pad_bytes=pad_bytes, lds_block_bytes=ROW_BYTES, pad_mode=pad_mode) if pad_mode else {}

    b = KernelBuilder("gemm_macrotile_mod", KERNEL_NAME, target=MCPU, wave_size=WAVE_SIZE)
    b.set_grid_dims(WG_GRID_M, WG_GRID_N, 1)
    b.set_block_dims(WAVE_SIZE)
    b.set_cluster_dims(CLUSTER_M, CLUSTER_N, 1)
    a_ptr, b_ptr, c_ptr = b.add_and_load_ptr_args([AccessKind.ReadOnly, AccessKind.ReadOnly, AccessKind.WriteOnly])

    d0 = ir.AffineExpr.get_dim(0)
    bx, by = b.block_id("x"), b.block_id("y")
    cgm, cm = b.delinearize_index(bx, (CLUSTER_GRID_M, CLUSTER_M))
    cgn, cn = b.delinearize_index(by, (CLUSTER_GRID_N, CLUSTER_N))
    # Per-WG global A/B/C offsets via tiled global tensors + slice: the cluster
    # hierarchy lives in the layout (M -> per-WG MACRO_TILE x CLUSTER WGs x grid)
    # so slicing the cluster axes folds cgm*WG_TILE + cm*MACRO_TILE into the offset
    # (no hand-rolled affine). The K axis is sliced per iteration for the TDM.
    A_TILED = tile(
        Layout((M, K), (stride_a, 2)),
        tile_sizes=((MACRO_TILE_M, CLUSTER_M), (MACRO_TILE_K,)),
        axes=((cm_ax, cgm_ax), (global_k,)),
    )
    B_TILED = tile(
        Layout((N, K), (stride_b, 2)),
        tile_sizes=((MACRO_TILE_N, CLUSTER_N), (MACRO_TILE_K,)),
        axes=((cn_ax, cgn_ax), (global_k,)),
    )
    C_TILED = tile(
        Layout((M, N), (stride_c, 4)),
        tile_sizes=((MACRO_TILE_M, CLUSTER_M), (MACRO_TILE_N, CLUSTER_N)),
        axes=((cm_ax, cgm_ax), (cn_ax, cgn_ax)),
    )
    TA = b.slice(Tensor(a_ptr, layout=A_TILED), {cm_ax: cm, cgm_ax: cgm})
    TB = b.slice(Tensor(b_ptr, layout=B_TILED), {cn_ax: cn, cgn_ax: cgn})
    TC = b.slice(Tensor(c_ptr, layout=C_TILED), {cm_ax: cm, cgm_ax: cgm, cn_ax: cn, cgn_ax: cgn})

    # Note: we are computing A . B^T, so we read A and B in row-major order.
    thread_layout, value_layout = WMMA_INTRINSIC.operand_read_layout(padded_row)
    tc_dsr_a = b.make_tiled_copy_descriptor(ds_load_b128, thread_layout=thread_layout, value_layout=value_layout)
    tc_dsr_b = b.make_tiled_copy_descriptor(ds_load_b128, thread_layout=thread_layout, value_layout=value_layout)

    # TDM uses padding for bank conflicts on gfx1250, no swizzling. 
    LDS_A_READ_TILED = tile(
        Layout((SUB_M, frag_bytes), (frag_bytes, 1)),
        tile_sizes=(1, frag_bytes),
        axes=(m_sub,),
    )
    LDS_B_READ_TILED = tile(
        Layout((SUB_N, frag_bytes), (frag_bytes, 1)),
        tile_sizes=(1, frag_bytes),
        axes=(n_sub,),
    )

    def _join_b128(lo, hi):
        # 2x b128 (vx4) -> vx8 = 16 f16/lane, K-block 0 then K-block 1.
        return b._make_register_range(list(b.split_register_range(lo, 4)) + list(b.split_register_range(hi, 4)))

    c0, c1 = b.constant_index(0), b.constant_index(1)

    accs_final = []
    n_accs = SUB_M * SUB_N
    acc_inits = [b.init_vgprx8(b.constant_i32(0)) for _ in range(n_accs)]

    @b.loop(c0, b.constant_index(k_tiles), c1, iter_args=acc_inits, results=accs_final)
    def _(k_iv, *accs):
        accs = list(accs)

        # TDM multicast recipient masks (group1 word0 [15:0], bit i = cluster WG
        # flat-id i = cm + cn*4). A is shared by the 4 cluster-N siblings (same cm)
        # -> 0x1111<<cm; B by the 4 cluster-M siblings (same cn) -> 0x000F<<(cn*4).
        # One HBM fetch per sibling group serves all 4 -> physical reuse-4.
        # Built inside the loop body (value-semantic SGPRs must not cross scf.for).
        if cluster_multicast:
            mc_a = b.s_lshl_b32(b.s_mov_b32(0x1111), b.index_to_sgpr(cm))
            mc_b = b.s_lshl_b32(b.s_mov_b32(0x000F), b.index_to_sgpr(b.affine_apply(d0 * 4, [cn])))
        else:
            mc_a = mc_b = 0

        with b.stage(STAGE_LOAD):
            lds_a_h, sA = b.alloc_lds_tensor(lds_total_a, layout=LDS_A_READ_TILED)
            lds_b_h, sB = b.alloc_lds_tensor(lds_total_b, layout=LDS_B_READ_TILED)

            # If using multicast, it is better to synchronize the cluster first.
            # Otherwise, if workgroups are too loosely coupled, the TDM risks
            # dropping to unicast for stragglers.
            if cluster_multicast:
                b.barrier(scope=BarrierScope.Cluster)

            # TDM A: cluster_multicast broadcasts this tile to the cluster-N
            # siblings via the recipient mask in the descriptor (one HBM fetch
            # serves all 4), then a cluster-scope barrier aligns the cluster.
            ta = b.slice(TA, {global_k: k_iv})
            a_g0, a_g1, a_g2, a_g3 = b.make_tdm_descriptor(
                lds_offset=sA.ptr,
                global_addr=a_ptr,
                global_byte_offset=ta.offset,
                element_bytes=2,
                dim0=K,
                dim1=MACRO_TILE_M,
                tile0=MACRO_TILE_K,
                tile1=MACRO_TILE_M,
                stride=K,
                recipient_mask=mc_a,
                **tdm_pad,
            )
            a_tdm_tok = b.tensor_load_to_lds(a_g0, a_g1, a_g2, a_g3)

            # TDM B: broadcast to the cluster-M siblings (recipient mask mc_b).
            tb = b.slice(TB, {global_k: k_iv})
            b_g0, b_g1, b_g2, b_g3 = b.make_tdm_descriptor(
                lds_offset=sB.ptr,
                global_addr=b_ptr,
                global_byte_offset=tb.offset,
                element_bytes=2,
                dim0=K,
                dim1=MACRO_TILE_N,
                tile0=MACRO_TILE_K,
                tile1=MACRO_TILE_N,
                stride=K,
                recipient_mask=mc_b,
                **tdm_pad,
            )
            b_tdm_tok = b.tensor_load_to_lds(b_g0, b_g1, b_g2, b_g3)

        with b.stage(STAGE_WRITE):
            # wait_deps drained tensorcnt (memory fence); the cluster barrier is
            # execution-only alignment of the cluster, so a plain barrier suffices.
            b.wait_deps_gfx1250(a_tdm_tok, b_tdm_tok)
            if cluster_multicast:
                b.barrier(scope=BarrierScope.Cluster)
            else:
                b.barrier(scope=BarrierScope.Workgroup)

        with b.stage(STAGE_READ):
            a_frags = b.transfer_tiles(sA, tc_dsr_a, unroll_axes=(m_sub,))
            b.wait_deps_gfx1250(a_frags)
            a_v = [_join_b128(a_frags.data_at((mi, 0)), a_frags.data_at((mi, 1))) for mi in range(SUB_M)]

        with b.stage(STAGE_COMPUTE):
            b_frags = b.transfer_tiles(sB, tc_dsr_b, unroll_axes=(n_sub,))
            b.wait_deps_gfx1250(b_frags)
            for ni in range(SUB_N):
                b_frag = _join_b128(b_frags.data_at((ni, 0)), b_frags.data_at((ni, 1)))
                for mi in range(SUB_M):
                    accs[mi * SUB_N + ni] = b.mfma(WMMA_MNEMONIC, accs[mi * SUB_N + ni], a_v[mi], b_frag)
            b.dealloc_lds(lds_a_h)
            b.dealloc_lds(lds_b_h)
        return accs

    flat_regs = []
    for acc in accs_final:
        flat_regs += list(b.split_register_range(acc, 8))

    GLOBAL_STORE_TILE_C = WMMA_INTRINSIC.output_tile_layout(stride_c)
    GLOBAL_STORE_SUB_TILE_C = Layout((SUB_M, SUB_N), (WMMA_TILE_M * stride_c, WMMA_TILE_N * 4))
    b.store_multi_fragment_to_global(
        b._make_register_range(flat_regs),
        c_ptr,
        TC.offset,
        GLOBAL_STORE_TILE_C,
        GLOBAL_STORE_SUB_TILE_C,
        b.global_store_b32,
        nt=False,
    )
    return b.build()


def _run_macrotile(*, cluster_multicast=False, pad_mode=None, print_opts=None):
    """Compile, assemble, and (if a gfx1250 GPU is present) execute + check."""
    np.random.seed(42)
    A = (np.random.randn(M, K) * 0.1).astype(np.float16)
    B = (np.random.randn(N, K) * 0.1).astype(np.float16)
    C_output = np.zeros(M * N, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = _build_macrotile(M, N, K, cluster_multicast=cluster_multicast, pad_mode=pad_mode)
        asm = compile_mlir_module_to_asm(
            module,
            pass_pipeline=make_default_pass_pipeline(PipelineConfig()),
            print_opts=print_opts,
        )

    path = assemble_to_hsaco(asm, target=MCPU, wavefront_size=WAVE_SIZE)
    if path is None:
        return "no-assembler", asm

    with hsaco_file(path):
        if not system_has_mcpu(mcpu=MCPU):
            return "no-gpu", asm
        execute_hsaco(
            hsaco_path=path,
            kernel_name=KERNEL_NAME,
            arguments=[
                InputArray(A.flatten().view(np.uint16)),
                InputArray(B.flatten().view(np.uint16)),
                OutputArray(C_output),
            ],
            grid_dim=(WG_GRID_M, WG_GRID_N, 1),
            block_dim=(WAVE_SIZE, 1, 1),
        )

    expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
    np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
    return "ok", asm


class TestMacrotileGfx1250:
    # TDM LDS-padding descriptor word (group1 word0): None = data_size only;
    # Padding = PaddingMode bits (block-interval pad); Iterate = TensorIterateMode
    # + group2 per-iteration LDS stride. Both pad LDS rows +16B (no write swizzle).
    _WORD0 = {None: 0x10000, TdmPadMode.Padding: 0x6D10000, TdmPadMode.Iterate: 0x90000}

    @pytest.mark.parametrize("pad_mode", [None, TdmPadMode.Padding, TdmPadMode.Iterate])
    @pytest.mark.parametrize("cluster_multicast", [False, True])
    def test_assembles(self, cluster_multicast, pad_mode):
        # The descriptor immediates are optimized out of the final asm, so prove the
        # padding mechanism is wired by checking group1 word0 in the IR.
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            mod_ir = str(_build_macrotile(M, N, K, cluster_multicast=cluster_multicast, pad_mode=pad_mode))
        word0 = self._WORD0[pad_mode]
        assert str(word0) in mod_ir, f"group1 word0 {hex(word0)} not in IR ({pad_mode})"

        # cluster_multicast wires the per-WG recipient mask (0x1111<<cm / 0x000F<<cn*4)
        # into group1 word0 [15:0] via a runtime shift + OR.
        if cluster_multicast:
            assert "s_or_b32" in mod_ir, "recipient-mask OR not in IR"
            assert "s_lshl_b32" in mod_ir, "recipient-mask shift not in IR"
        else:
            assert "s_or_b32" not in mod_ir, "unexpected mask wiring without multicast"

        # Assembly is the primary gate on a no-GPU platform; a missing assembler
        # is a real failure, not a skip (skipping would green-light zero validation).
        status, _ = _run_macrotile(cluster_multicast=cluster_multicast, pad_mode=pad_mode)
        assert status != "no-assembler", f"{MCPU} HSACO did not assemble"
        if status == "no-gpu":
            pytest.skip(f"{MCPU} GPU not available; HSACO assembled OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_multicast", action="store_true")
    parser.add_argument("--pad-mode", choices=["padding", "iterate"], default=None)
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--print-ir-after-all", action="store_true")
    args = parser.parse_args()

    from aster.compiler.core import PrintOptions

    status, asm = _run_macrotile(
        cluster_multicast=args.cluster_multicast,
        pad_mode=args.pad_mode,
        print_opts=PrintOptions.from_flags(
            print_ir_after_all=args.print_ir_after_all,
            print_asm=args.print_asm,
        ),
    )
    if args.print_asm:
        print(asm)
    path = "cluster_multicast" if args.cluster_multicast else "phase1"
    if status == "no-assembler":
        print(f"FAIL: {MCPU} HSACO did not assemble ({path})")
        sys.exit(1)
    elif status == "no-gpu":
        print(f"{MCPU} GPU not available; HSACO assembled OK ({path})")
    else:
        print(f"PASS: macrotile GEMM {M}x{N}x{K} matches reference ({path})")
