import subprocess
import sys
import textwrap

import numpy as np
import pytest

from aster import ir
from aster.layout import Layout
from aster.dialects.kernel_builder import KernelBuilder, MFMA_F16_CDNA4, MfmaConfig
from aster.dialects.amdgcn import AccessKind
from aster.compiler.core import compile_mlir_module_to_asm, assemble_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray
from aster.execution.helpers import hsaco_file
from aster.execution.utils import system_has_mcpu

from aster.pass_pipelines import (
    builtin_module,
    phase_amdgcn_backend,
    phase_nop_insertion,
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
)

MCPUS = ["gfx942", "gfx950"]
N_THREADS = 64
ILP_LEVELS = [-1, 0, 1, 2]

MFMA_F16_CDNA3 = MfmaConfig(
    opcode="v_mfma_f32_16x16x16_f16",
    shape=(16, 16, 16),
    a_regs=2,
    b_regs=2,
    c_regs=4,
)
ELEM_BYTES = 8
DWORD_BYTES = 4
ACC_BASE_BYTES = N_THREADS * ELEM_BYTES

LINEAR = Layout(sizes=N_THREADS, strides=ELEM_BYTES)
LINEAR_DWORD = Layout(sizes=N_THREADS, strides=DWORD_BYTES)

K_TILES = 4
LDS_SLOT_BYTES = 512
STAGE_BYTES = 2 * LDS_SLOT_BYTES
STAGE_LAYOUT = Layout(sizes=(2, N_THREADS), strides=(STAGE_BYTES, ELEM_BYTES))


def _mfma_cfg(mcpu: str) -> MfmaConfig:
    return MFMA_F16_CDNA4 if mcpu == "gfx950" else MFMA_F16_CDNA3


def _kernel_name(rung_name: str, mcpu: str) -> str:
    return f"{rung_name}_{mcpu}"


def _mfma(b, acc, a_vx2, b_vx2, mcpu: str):
    cfg = _mfma_cfg(mcpu)
    if cfg.reads_vx4:
        frag = b.join_vx2_to_vx4(a_vx2, b_vx2)
        return b.mfma(cfg.opcode, acc, frag, frag)
    return b.mfma(cfg.opcode, acc, a_vx2, b_vx2)


def ilp_sched_pipeline(ll_ilp_sched: int) -> str:
    return builtin_module(
        PHASE_PRE_SCHEDULING_CLEANUP,
        PHASE_SCHEDULING,
        PHASE_POST_SCHEDULING_CLEANUP,
        PHASE_SROA,
        POST_SROA_CLEANUPS,
        PHASE_AFFINE_EXPANSION,
        PHASE_SROA,
        POST_SROA_CLEANUPS,
        PHASE_EXPAND_MD_OPS,
        PHASE_LOWER_TO_AMDGCN,
        phase_amdgcn_backend(ll_ilp_sched=ll_ilp_sched),
        phase_nop_insertion(delays=0),
    )


def _begin(name, mcpu, lds_bytes=0):
    kernel = _kernel_name(name, mcpu)
    b = KernelBuilder(f"{kernel}_mod", kernel, target=mcpu)
    if lds_bytes:
        b.set_shared_memory_size(lds_bytes)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    src_ptr, dst_ptr = b.load_args()
    num_records = b.s_mov_b32(65536)
    soffset = b.s_mov_b32(0)
    src_rsrc = b.make_buffer_rsrc(src_ptr, num_records, b.constant_i32(0))
    dst_rsrc = b.make_buffer_rsrc(dst_ptr, num_records, b.constant_i32(0))
    return b, src_rsrc, dst_rsrc, soffset


def _thread_voff(b, tid, layout=LINEAR):
    return b.index_to_vgpr(b.layout_apply(tid, layout))


def _lds_voff(b, lds_base, tid):
    return b.index_to_vgpr(b.layout_sum(lds_base, b.layout_apply(tid, LINEAR)))


def _acc_voff(b, tid):
    return b.index_to_vgpr(
        b.layout_sum(
            b.constant_index(ACC_BASE_BYTES), b.layout_apply(tid, LINEAR_DWORD)
        )
    )


def _stage_voff(b, lds_base, parity, tid):
    return b.index_to_vgpr(
        b.layout_sum(lds_base, b.layout_apply((parity, tid), STAGE_LAYOUT))
    )


def _live_mfma(b, a_frag, b_frag, dst_rsrc, tid, soffset, mcpu):
    acc = b.init_agprx4(b.constant_i32(0))
    acc = _mfma(b, acc, a_frag, b_frag, mcpu)
    a0, _a1, _a2, _a3 = b.split_ax4(acc)
    b.buffer_store_dword(a0, dst_rsrc, soffset, _acc_voff(b, tid))


def build_r0_register_only(name, mcpu):
    b, src_rsrc, dst_rsrc, soffset = _begin(name, mcpu)
    tid = b.thread_id("x")
    voff = _thread_voff(b, tid, LINEAR_DWORD)
    v = b.buffer_load(src_rsrc, soffset, voff)
    b.wait_vmcnt(0)
    b.buffer_store_dword(v, dst_rsrc, soffset, voff)
    return b.build()


def build_r1_global_load_store(name, mcpu):
    b, src_rsrc, dst_rsrc, soffset = _begin(name, mcpu)
    tid = b.thread_id("x")
    voff = _thread_voff(b, tid)
    data = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
    b.wait_vmcnt(0)
    b.buffer_store_dwordx2(data, dst_rsrc, soffset, voff)
    return b.build()


def build_r2_lds_roundtrip_mfma(name, mcpu):
    b, src_rsrc, dst_rsrc, soffset = _begin(
        name, mcpu, lds_bytes=N_THREADS * ELEM_BYTES
    )
    tid = b.thread_id("x")
    voff = _thread_voff(b, tid)
    data = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
    b.wait_vmcnt(0)
    tok_w = b.ds_write_b64(data, voff)
    b.wait_deps(tok_w)
    a_frag, tok_r = b.ds_read_b64(voff)
    b.wait_deps(tok_r)
    b_frag, tok_r2 = b.ds_read_b64(voff)
    b.wait_deps(tok_r2)
    b.buffer_store_dwordx2(a_frag, dst_rsrc, soffset, voff)
    _live_mfma(b, a_frag, b_frag, dst_rsrc, tid, soffset, mcpu)
    return b.build()


def build_r3_lds_roundtrip_barrier_mfma(name, mcpu):
    b, src_rsrc, dst_rsrc, soffset = _begin(
        name, mcpu, lds_bytes=N_THREADS * ELEM_BYTES
    )
    tid = b.thread_id("x")
    voff = _thread_voff(b, tid)
    data = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
    b.wait_vmcnt(0)
    tok_w = b.ds_write_b64(data, voff)
    b.wait_deps(tok_w)
    b.s_barrier()
    a_frag, tok_r = b.ds_read_b64(voff)
    b.wait_deps(tok_r)
    b_frag, tok_r2 = b.ds_read_b64(voff)
    b.wait_deps(tok_r2)
    b.buffer_store_dwordx2(a_frag, dst_rsrc, soffset, voff)
    _live_mfma(b, a_frag, b_frag, dst_rsrc, tid, soffset, mcpu)
    return b.build()


def build_r4_lds_kloop_singlebuf_multiop(name, mcpu):
    b, src_rsrc, dst_rsrc, soffset = _begin(name, mcpu, lds_bytes=2 * LDS_SLOT_BYTES)
    tid = b.thread_id("x")
    voff = _thread_voff(b, tid)
    lds_h, lds_off = b.alloc_lds(2 * LDS_SLOT_BYTES)
    lds_pt = _lds_voff(b, lds_off, tid)
    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    ck = b.constant_index(K_TILES)
    acc_init = b.init_agprx4(b.constant_i32(0))
    res = []

    @b.loop(c0, ck, c1, iter_args=[acc_init], results=res)
    def _(k_iv, acc):
        dA = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
        dB = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
        b.wait_vmcnt(0)
        tw1 = b.ds_write_b64(dA, lds_pt, const_offset=b.constant_i32(0))
        tw2 = b.ds_write_b64(dB, lds_pt, const_offset=b.constant_i32(LDS_SLOT_BYTES))
        b.wait_deps(tw1, tw2)
        a_frag, tr1 = b.ds_read_b64(lds_pt, const_offset=b.constant_i32(0))
        b_frag, tr2 = b.ds_read_b64(lds_pt, const_offset=b.constant_i32(LDS_SLOT_BYTES))
        b.wait_deps(tr1, tr2)
        b.buffer_store_dwordx2(a_frag, dst_rsrc, soffset, voff)
        acc = _mfma(b, acc, a_frag, b_frag, mcpu)
        return [acc]

    [acc_final] = res
    a0, _a1, _a2, _a3 = b.split_ax4(acc_final)
    b.buffer_store_dword(a0, dst_rsrc, soffset, _acc_voff(b, tid))
    b.dealloc_lds(lds_h)
    return b.build()


def build_r5_lds_kloop_doublebuf_rotate(name, mcpu):
    b, src_rsrc, dst_rsrc, soffset = _begin(name, mcpu, lds_bytes=2 * STAGE_BYTES)
    tid = b.thread_id("x")
    voff = _thread_voff(b, tid)
    lds_h, lds_off = b.alloc_lds(2 * STAGE_BYTES)
    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    ck = b.constant_index(K_TILES)
    acc_init = b.init_agprx4(b.constant_i32(0))
    res = []
    d0 = ir.AffineExpr.get_dim(0)

    @b.loop(c0, ck, c1, iter_args=[acc_init], results=res)
    def _(k_iv, acc):
        parity = b.affine_apply(d0 % 2, [k_iv])
        stage_v = _stage_voff(b, lds_off, parity, tid)
        dA = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
        dB = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
        b.wait_vmcnt(0)
        tw1 = b.ds_write_b64(dA, stage_v, const_offset=b.constant_i32(0))
        tw2 = b.ds_write_b64(dB, stage_v, const_offset=b.constant_i32(LDS_SLOT_BYTES))
        b.wait_deps(tw1, tw2)
        a_frag, tr1 = b.ds_read_b64(stage_v, const_offset=b.constant_i32(0))
        b_frag, tr2 = b.ds_read_b64(
            stage_v, const_offset=b.constant_i32(LDS_SLOT_BYTES)
        )
        b.wait_deps(tr1, tr2)
        b.buffer_store_dwordx2(a_frag, dst_rsrc, soffset, voff)
        acc = _mfma(b, acc, a_frag, b_frag, mcpu)
        return [acc]

    [acc_final] = res
    a0, _a1, _a2, _a3 = b.split_ax4(acc_final)
    b.buffer_store_dword(a0, dst_rsrc, soffset, _acc_voff(b, tid))
    b.dealloc_lds(lds_h)
    return b.build()


def build_r6_g2s_vmcnt_to_lds(name, mcpu):
    b, src_rsrc, dst_rsrc, soffset = _begin(name, mcpu, lds_bytes=2 * LDS_SLOT_BYTES)
    tid = b.thread_id("x")
    voff = _thread_voff(b, tid, LINEAR_DWORD)
    lds_h, lds_off = b.alloc_lds(2 * LDS_SLOT_BYTES)
    lds_pt = _lds_voff(b, lds_off, tid)
    m0 = b.alloc_m0()
    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    ck = b.constant_index(K_TILES)
    acc_init = b.init_agprx4(b.constant_i32(0))
    res = []

    @b.loop(c0, ck, c1, iter_args=[acc_init], results=res)
    def _(k_iv, acc):
        b.set_m0(m0, b.constant_i32(0))
        g1 = b.g2s_buffer_load_dword(m0, src_rsrc, soffset, voff)
        b.set_m0(m0, b.constant_i32(LDS_SLOT_BYTES))
        g2 = b.g2s_buffer_load_dword(m0, src_rsrc, soffset, voff)
        b.wait_deps(g1, g2)
        a_frag, tr1 = b.ds_read_b64(lds_pt, const_offset=b.constant_i32(0))
        b_frag, tr2 = b.ds_read_b64(lds_pt, const_offset=b.constant_i32(LDS_SLOT_BYTES))
        b.wait_deps(tr1, tr2)
        acc = _mfma(b, acc, a_frag, b_frag, mcpu)
        return [acc]

    [acc_final] = res
    a0, _a1, _a2, _a3 = b.split_ax4(acc_final)
    b.buffer_store_dword(a0, dst_rsrc, soffset, _acc_voff(b, tid))
    b.dealloc_lds(lds_h)
    return b.build()


def build_r7_two_lds_buffers_control(name, mcpu):
    b, src_rsrc, dst_rsrc, soffset = _begin(name, mcpu, lds_bytes=2 * LDS_SLOT_BYTES)
    tid = b.thread_id("x")
    voff = _thread_voff(b, tid)
    lds_a_h, lds_a = b.alloc_lds(LDS_SLOT_BYTES)
    lds_b_h, lds_b = b.alloc_lds(LDS_SLOT_BYTES)
    lds_a_pt = _lds_voff(b, lds_a, tid)
    lds_b_pt = _lds_voff(b, lds_b, tid)
    c0 = b.constant_index(0)
    c1 = b.constant_index(1)
    ck = b.constant_index(K_TILES)
    acc_init = b.init_agprx4(b.constant_i32(0))
    res = []

    @b.loop(c0, ck, c1, iter_args=[acc_init], results=res)
    def _(k_iv, acc):
        dA = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
        dB = b.buffer_load_dwordx2(src_rsrc, soffset, voff)
        b.wait_vmcnt(0)
        tw1 = b.ds_write_b64(dA, lds_a_pt)
        tw2 = b.ds_write_b64(dB, lds_b_pt)
        b.wait_deps(tw1, tw2)
        a_frag, tr1 = b.ds_read_b64(lds_a_pt)
        b_frag, tr2 = b.ds_read_b64(lds_b_pt)
        b.wait_deps(tr1, tr2)
        b.buffer_store_dwordx2(a_frag, dst_rsrc, soffset, voff)
        acc = _mfma(b, acc, a_frag, b_frag, mcpu)
        return [acc]

    [acc_final] = res
    a0, _a1, _a2, _a3 = b.split_ax4(acc_final)
    b.buffer_store_dword(a0, dst_rsrc, soffset, _acc_voff(b, tid))
    b.dealloc_lds(lds_a_h)
    b.dealloc_lds(lds_b_h)
    return b.build()


RUNGS = {
    "r0_register_only": dict(
        build=build_r0_register_only,
        enters_ilp=False,
        repro_suspect=False,
        identity=False,
    ),
    "r1_global_load_store": dict(
        build=build_r1_global_load_store,
        enters_ilp=False,
        repro_suspect=False,
        identity=True,
    ),
    "r2_lds_roundtrip_mfma": dict(
        build=build_r2_lds_roundtrip_mfma,
        enters_ilp=True,
        repro_suspect=False,
        identity=True,
    ),
    "r3_lds_roundtrip_barrier_mfma": dict(
        build=build_r3_lds_roundtrip_barrier_mfma,
        enters_ilp=True,
        repro_suspect=False,
        identity=True,
    ),
    "r4_lds_kloop_singlebuf_multiop": dict(
        build=build_r4_lds_kloop_singlebuf_multiop,
        enters_ilp=True,
        repro_suspect=True,
        identity=True,
    ),
    "r5_lds_kloop_doublebuf_rotate": dict(
        build=build_r5_lds_kloop_doublebuf_rotate,
        enters_ilp=True,
        repro_suspect=True,
        identity=True,
    ),
    "r6_g2s_vmcnt_to_lds": dict(
        build=build_r6_g2s_vmcnt_to_lds,
        enters_ilp=True,
        repro_suspect=True,
        identity=False,
    ),
    "r7_two_lds_buffers_control": dict(
        build=build_r7_two_lds_buffers_control,
        enters_ilp=True,
        repro_suspect=False,
        identity=True,
    ),
}


def _compile_rung(rung_name, ll_ilp_sched, mcpu):
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = RUNGS[rung_name]["build"](rung_name, mcpu)
        return compile_mlir_module_to_asm(
            module, pass_pipeline=ilp_sched_pipeline(ll_ilp_sched)
        )


def _ref_inputs():
    src = np.zeros(
        (ACC_BASE_BYTES + N_THREADS * DWORD_BYTES) // DWORD_BYTES, dtype=np.float32
    )
    for tid in range(N_THREADS):
        for j in range(ELEM_BYTES // DWORD_BYTES):
            src[LINEAR(tid) // DWORD_BYTES + j] = (
                tid * (ELEM_BYTES // DWORD_BYTES) + j + 1.0
            )
    return src


def _run_gpu(rung_name, ll_ilp_sched, mcpu):
    asm = _compile_rung(rung_name, ll_ilp_sched, mcpu)
    path = assemble_to_hsaco(asm, target=mcpu, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler lacks {mcpu} support")
    src = _ref_inputs()
    dst = np.zeros(
        (ACC_BASE_BYTES + N_THREADS * DWORD_BYTES) // DWORD_BYTES, dtype=np.float32
    )
    with hsaco_file(path):
        if not system_has_mcpu(mcpu=mcpu):
            pytest.skip(f"{mcpu} GPU not available")
        execute_hsaco(
            hsaco_path=path,
            kernel_name=_kernel_name(rung_name, mcpu),
            arguments=[InputArray(src), OutputArray(dst)],
            grid_dim=(1, 1, 1),
            block_dim=(N_THREADS, 1, 1),
        )
    return dst, src


@pytest.mark.parametrize("mcpu", MCPUS)
@pytest.mark.parametrize("rung_name", list(RUNGS))
@pytest.mark.parametrize("ll_ilp_sched", ILP_LEVELS)
def test_rung_compiles(mcpu, rung_name, ll_ilp_sched):
    asm = _compile_rung(rung_name, ll_ilp_sched, mcpu)
    assert asm and len(asm) > 0
    path = assemble_to_hsaco(asm, target=mcpu, wavefront_size=64)
    if path is None:
        pytest.skip(f"LLVM assembler lacks {mcpu} support")


_DUMP_DRIVER = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, {test_dir!r})
    import test_stress_test_ilp as t
    t._compile_rung({rung!r}, {level}, {mcpu!r})
    """
)


def _scheduler_trace(rung_name, ll_ilp_sched, mcpu):
    import os

    test_dir = os.path.dirname(os.path.abspath(__file__))
    code = _DUMP_DRIVER.format(
        test_dir=test_dir, rung=rung_name, level=ll_ilp_sched, mcpu=mcpu
    )
    env = dict(os.environ)
    env["ASTER_ILP_DUMP_GRAPH"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"
    return proc.stderr


_ILP_RUNGS = [n for n, m in RUNGS.items() if m["enters_ilp"]]


@pytest.mark.parametrize("mcpu", MCPUS)
@pytest.mark.parametrize("rung_name", _ILP_RUNGS)
@pytest.mark.parametrize("ll_ilp_sched", [0, 1, 2])
def test_mfma_block_reaches_ilp_path(mcpu, rung_name, ll_ilp_sched):
    trace = _scheduler_trace(rung_name, ll_ilp_sched, mcpu)
    assert "ILP-SCHED-GRAPH" in trace, f"no ILP graph dumped:\n{trace}"
    assert "[XDL]" in trace, f"no MFMA node in any scheduled block:\n{trace}"
    solved = "ILP-WINDOW" in trace or "ILP-WHOLE" in trace
    assert solved, f"MFMA block did not reach CP-SAT solve:\n{trace}"


_LOOP_RUNGS = [
    "r4_lds_kloop_singlebuf_multiop",
    "r5_lds_kloop_doublebuf_rotate",
    "r7_two_lds_buffers_control",
]


@pytest.mark.parametrize("mcpu", MCPUS)
@pytest.mark.parametrize("rung_name", _LOOP_RUNGS)
@pytest.mark.parametrize("ll_ilp_sched", ILP_LEVELS)
def test_loop_body_intra_iter_raw_preserved(mcpu, rung_name, ll_ilp_sched):
    asm = _compile_rung(rung_name, ll_ilp_sched, mcpu)
    ds = [
        line.strip()
        for line in asm.splitlines()
        if "ds_write" in line or "ds_read" in line
    ]
    writes = [i for i, line in enumerate(ds) if "ds_write" in line]
    reads = [i for i, line in enumerate(ds) if "ds_read" in line]
    assert writes and reads, f"expected ds writes and reads in {rung_name}:\n{asm}"
    assert max(writes) < min(reads), (
        f"{rung_name} @ {mcpu} ll_ilp_sched={ll_ilp_sched}: a ds_read was scheduled "
        f"before a ds_write inside the loop body (intra-iteration RAW broken)"
    )


@pytest.mark.parametrize("mcpu", MCPUS)
@pytest.mark.parametrize("rung_name", list(RUNGS))
@pytest.mark.parametrize("ll_ilp_sched", ILP_LEVELS)
def test_rung_executes(mcpu, rung_name, ll_ilp_sched):
    dst, src = _run_gpu(rung_name, ll_ilp_sched, mcpu)
    if RUNGS[rung_name]["identity"]:
        np.testing.assert_array_equal(
            dst[: ACC_BASE_BYTES // DWORD_BYTES],
            src[: ACC_BASE_BYTES // DWORD_BYTES],
            err_msg=(
                f"{rung_name} @ {mcpu} ll_ilp_sched={ll_ilp_sched}: identity copy "
                f"corrupted (likely a reordered cross-iteration LDS write/read)"
            ),
        )


@pytest.mark.parametrize("mcpu", MCPUS)
@pytest.mark.parametrize(
    "rung_name", [n for n, m in RUNGS.items() if m["repro_suspect"]]
)
def test_repro_ilp_vs_greedy(mcpu, rung_name):
    greedy_dst, src = _run_gpu(rung_name, -1, mcpu)
    ilp_dst, _ = _run_gpu(rung_name, 2, mcpu)
    if RUNGS[rung_name]["identity"]:
        np.testing.assert_array_equal(
            ilp_dst[: ACC_BASE_BYTES // DWORD_BYTES],
            greedy_dst[: ACC_BASE_BYTES // DWORD_BYTES],
            err_msg=f"{rung_name} @ {mcpu}: ILP output diverges from greedy (illegal reorder)",
        )
