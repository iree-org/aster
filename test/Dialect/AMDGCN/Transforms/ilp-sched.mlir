// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=0 ilp-time-limit-ms=5000})))" --split-input-file | FileCheck %s --check-prefix=L0
// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=1 ilp-time-limit-ms=2000})))" --split-input-file | FileCheck %s --check-prefix=L1
// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=2 ilp-time-limit-ms=2000})))" --split-input-file | FileCheck %s --check-prefix=L2
// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=2 mfma-gap=2 lgkm-gap=2 ilp-time-limit-ms=2000})))" --split-input-file | FileCheck %s --check-prefix=INTL
// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=0 ilp-max-load-distance=3 ilp-time-limit-ms=2000})))" --split-input-file | FileCheck %s --check-prefix=LOADDIST

!v = !amdgcn.vgpr

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // B1 scaffold: the ILP scheduler (level 0, no model yet) preserves source
  // order. Confirms the pass registers, parses its options, and runs as a
  // no-op identity reorder without disturbing the IR.
  // L0-LABEL: kernel @identity
  // L0:         alloca
  // L0:         alloca
  // L0:         alloca
  // L0:         end_kernel
  amdgcn.kernel @identity {
    %v0 = amdgcn.alloca : !v
    %v1 = amdgcn.alloca : !v
    %v2 = amdgcn.alloca : !v
    amdgcn.end_kernel
  }
}

// -----

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // Level 0 (correctness): the CP-SAT model must respect the ds_read -> mfma
  // data dependency (each ds_read produces a fragment the mfma consumes). The
  // hard precedence constraint keeps both reads before the mfma regardless of
  // the source-order-biased objective.
  // L0-LABEL: kernel @level0_respects_deps
  // L0:         ds_read_b64
  // L0:         ds_read_b64
  // L0:         v_mfma_f32_16x16x16_f16
  // L0:         end_kernel
  amdgcn.kernel @level0_respects_deps {
    %c0_0 = amdgcn.alloca : !v
    %c0_1 = amdgcn.alloca : !v
    %c0_2 = amdgcn.alloca : !v
    %c0_3 = amdgcn.alloca : !v
    %acc = amdgcn.make_register_range %c0_0, %c0_1, %c0_2, %c0_3 : !v, !v, !v, !v
    %fa0 = amdgcn.alloca : !v
    %fa1 = amdgcn.alloca : !v
    %fragA = amdgcn.make_register_range %fa0, %fa1 : !v, !v
    %fb0 = amdgcn.alloca : !v
    %fb1 = amdgcn.alloca : !v
    %fragB = amdgcn.make_register_range %fb0, %fb1 : !v, !v
    %lar = amdgcn.alloca : !v
    %lbr = amdgcn.alloca : !v
    %c0i = arith.constant 0 : i32
    %rdA, %rtA = amdgcn.ds_read_b64 dest %fragA addr %lar offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rdB, %rtB = amdgcn.ds_read_b64 dest %fragB addr %lbr offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rdA, %rdB, %acc)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    amdgcn.end_kernel
  }
}

// -----

// Level-b VMEM interleaving: 9 mutually-independent global_load_dword ops
// followed by 4 VALU fillers.
// level 0: source-order biased; all 9 loads survive (no ops dropped).
// level 1: default vmem-gap=0 -> no VMEM spacing constraint -> identical to
//          level 0.  Both levels interleave loads and v_mov freely under the
//          source-order tie-break; the exact pattern is an ILP solver artifact.

!v   = !amdgcn.vgpr
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // All 9 loads must survive; exact order is solver-determined.
  // L0-LABEL:    kernel @ilp_level_b_vmem
  // L0-COUNT-9:  global_load_dword

  // Level 1 with vmem-gap=0 is identical to level 0.
  // L1-LABEL:    kernel @ilp_level_b_vmem
  // L1-COUNT-9:  global_load_dword
  amdgcn.kernel @ilp_level_b_vmem {
    %dst  = amdgcn.alloca : !v
    %sa0  = amdgcn.alloca : !s
    %sa1  = amdgcn.alloca : !s
    %gaddr = amdgcn.make_register_range %sa0, %sa1 : !s, !s
    %off  = amdgcn.alloca : !v
    %c0i  = arith.constant 0 : i32
    // 9 mutually-independent VMEM loads, contiguous in source order. Tokens are
    // unused (no amdgcn.wait) so there are no serialization edges between them.
    %r0, %t0 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r1, %t1 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r2, %t2 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r3, %t3 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r4, %t4 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r5, %t5 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r6, %t6 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r7, %t7 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %r8, %t8 = amdgcn.global_load_dword dest %dst addr %gaddr offset d(%off) + c(%c0i)
        : outs(!v) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    // Independent VALU fillers (VMEM burst penalty = 0 for VALU). Their setup
    // (alloca + constants) supplies the non-VMEM ops the solver interleaves.
    %fa  = amdgcn.alloca : !v
    %cf0 = arith.constant 1 : i32
    %cf1 = arith.constant 2 : i32
    %cf2 = arith.constant 3 : i32
    %cf3 = arith.constant 4 : i32
    %fva = amdgcn.v_mov_b32 outs(%fa) ins(%cf0) : outs(!v) ins(i32)
    %fvb = amdgcn.v_mov_b32 outs(%fa) ins(%cf1) : outs(!v) ins(i32)
    %fvc = amdgcn.v_mov_b32 outs(%fa) ins(%cf2) : outs(!v) ins(i32)
    %fvd = amdgcn.v_mov_b32 outs(%fa) ins(%cf3) : outs(!v) ins(i32)
    amdgcn.end_kernel
  }
}

// -----

// Level-c load-latency test: one global_load_dwordx2 feeding ds_write_b64 (RAW),
// plus an independent ds_read_b64 + v_mfma (XDL, triggers ILP path) and a v_mov
// filler.  The ILP hoists both independent ops (ds_read + v_mov) into the global
// load's VMEM shadow before the ds_write can issue.  Level=2 with a single MFMA
// adds no mfma-gap constraint, so the schedule is identical to level=1.

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // ILP hoists the independent ds_read (the MFMA operand, on a distinct LDS
  // buffer) into the global_load VMEM shadow ahead of the RAW-blocked ds_write;
  // the makespan-neutral v_mov filler trails.  MFMA closes the block.
  // L1-LABEL:   kernel @ilp_level_c_hide_load
  //      L1:      global_load_dwordx2
  //      L1:      ds_read_b64
  //      L1:      ds_write_b64
  //      L1:      v_mov_b32
  //      L1:      v_mfma_f32_16x16x16_f16

  // Level=2 with one MFMA: no mfma-gap constraint fires -> same order as level=1.
  // L2-LABEL:   kernel @ilp_level_c_hide_load
  //      L2:      global_load_dwordx2
  //      L2:      ds_read_b64
  //      L2:      ds_write_b64
  //      L2:      v_mov_b32
  //      L2:      v_mfma_f32_16x16x16_f16
  amdgcn.kernel @ilp_level_c_hide_load {
    // global-load destination.
    %gd0 = amdgcn.alloca : !v
    %gd1 = amdgcn.alloca : !v
    %gdst = amdgcn.make_register_range %gd0, %gd1 : !v, !v

    // SGPR base address for global load.
    %sa0   = amdgcn.alloca : !s
    %sa1   = amdgcn.alloca : !s
    %gaddr = amdgcn.make_register_range %sa0, %sa1 : !s, !s
    %goff  = amdgcn.alloca : !v

    // Shared per-thread LDS address; distinct LDS buffers (distinct alloc_lds
    // offsets) so memdep does not order the write against the independent read.
    %perthread = amdgcn.alloca : !v
    %bufw = amdgcn.alloc_lds 256 alignment 16
    %bufr = amdgcn.alloc_lds 256 alignment 16
    %offw = amdgcn.get_lds_offset %bufw : i32
    %offr = amdgcn.get_lds_offset %bufr : i32

    // MFMA accumulator.
    %c0  = amdgcn.alloca : !v
    %c1  = amdgcn.alloca : !v
    %c2  = amdgcn.alloca : !v
    %c3  = amdgcn.alloca : !v
    %acc = amdgcn.make_register_range %c0, %c1, %c2, %c3 : !v, !v, !v, !v

    // Fragment destination for the ds_read that feeds the MFMA.
    %fa0  = amdgcn.alloca : !v
    %fa1  = amdgcn.alloca : !v
    %frag = amdgcn.make_register_range %fa0, %fa1 : !v, !v

    // Independent filler destination.
    %fd0 = amdgcn.alloca : !v

    %c0i = arith.constant 0 : i32

    // Long-latency global load; %loaded feeds ds_write below (RAW).
    %loaded, %ltok = amdgcn.global_load_dwordx2 dest %gdst addr %gaddr offset d(%goff) + c(%c0i)
        : outs(!vx2) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>

    // RAW consumer: blocked on %loaded -> ILP sinks it after independent ops.
    %wt = amdgcn.ds_write_b64 data %loaded addr %perthread offset c(%offw)
        : ins(!vx2, !v) mods(i32) -> !amdgcn.write_token<shared>

    // Independent ds_read (no dep on global load) -> hoisted into VMEM shadow.
    %rd, %rtok = amdgcn.ds_read_b64 dest %frag addr %perthread offset c(%offr)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>

    // MFMA: XDL op guarantees this block enters the ILP solver (not greedy).
    %mout = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %acc)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)

    // Independent VALU filler -> also hoisted into the global_load's VMEM shadow.
    %fv = amdgcn.v_mov_b32 outs(%fd0) ins(%c0i) : outs(!v) ins(i32)

    amdgcn.end_kernel
  }
}

// -----

// Mixed-queue K-step: VMEM (global_load + buffer_load), LGKM (ds_write +
// ds_read), XDL (two-op MFMA chain), and independent VALU (v_mov_b32).
// level 0/1: default gaps (vmem-gap=0, lgkm-gap=0, mfma-gap=0) -> no spacing
//            constraints; two MFMAs issue contiguously, ds_write and v_mov sink
//            below them.
// level 2: mfma-gap=4 forces >= 4 real ops between consecutive MFMAs;
//          the two ds_writes and two v_movs fill that gap.

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>
!sx4 = !amdgcn.sgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // ds_reads hoist first (no deps); the loads issue, then ds_writes and v_movs
  // fill the load shadow; the two MFMAs issue contiguously at the end.
  // L0-LABEL:   kernel @gemm_k_step
  // L0:           ds_read_b64
  // L0:           ds_read_b64
  // L0:           global_load_dwordx2
  // L0:           buffer_load_dwordx2
  // L0:           ds_write_b64
  // L0:           v_mov_b32
  // L0:           v_mfma_f32_16x16x16_f16
  // L0-NEXT:      v_mfma_f32_16x16x16_f16

  // Level 1 with default gaps (vmem-gap=0, lgkm-gap=0) is a no-op vs level 0.
  // L1-LABEL:   kernel @gemm_k_step
  // L1:           ds_read_b64
  // L1:           ds_read_b64
  // L1:           global_load_dwordx2
  // L1:           buffer_load_dwordx2
  // L1:           ds_write_b64
  // L1:           v_mov_b32
  // L1:           v_mfma_f32_16x16x16_f16
  // L1-NEXT:      v_mfma_f32_16x16x16_f16

  // mfma-gap=4: the two MFMAs are spaced apart; one ds_write + two v_movs fill
  // the gap between them (the other ds_write issues before the first MFMA).
  // Precedence: loads before their ds_writes; ds_reads before both MFMAs.
  // L2-LABEL:   kernel @gemm_k_step
  // L2:           ds_read_b64
  // L2:           ds_read_b64
  // L2:           v_mfma_f32_16x16x16_f16
  // L2:           ds_write_b64
  // L2:           v_mov_b32
  // L2:           v_mov_b32
  // L2:           v_mfma_f32_16x16x16_f16

  // ilp-max-load-distance=3: each load's consumer must issue within 3 ranks, so
  // the loads are pulled down toward their uses. Unlike level 0 (global_load
  // above both MFMAs), the MFMAs now precede global_load (its ds_read inputs
  // stay at top, the VMEM load drops next to its ds_write).
  // LOADDIST-LABEL:   kernel @gemm_k_step
  // LOADDIST:           ds_read_b64
  // LOADDIST:           v_mfma_f32_16x16x16_f16
  // LOADDIST:           v_mfma_f32_16x16x16_f16
  // LOADDIST:           global_load_dwordx2
  // LOADDIST:           ds_write_b64
  amdgcn.kernel @gemm_k_step {
    // C accumulator tiles.
    %c0_0 = amdgcn.alloca : !v
    %c0_1 = amdgcn.alloca : !v
    %c0_2 = amdgcn.alloca : !v
    %c0_3 = amdgcn.alloca : !v
    %acc0 = amdgcn.make_register_range %c0_0, %c0_1, %c0_2, %c0_3 : !v, !v, !v, !v
    %c1_0 = amdgcn.alloca : !v
    %c1_1 = amdgcn.alloca : !v
    %c1_2 = amdgcn.alloca : !v
    %c1_3 = amdgcn.alloca : !v
    %acc1 = amdgcn.make_register_range %c1_0, %c1_1, %c1_2, %c1_3 : !v, !v, !v, !v

    // global-load A: sx2 base + voffset -> vx2.
    %ga0 = amdgcn.alloca : !s
    %ga1 = amdgcn.alloca : !s
    %gA  = amdgcn.make_register_range %ga0, %ga1 : !s, !s
    %gAoff = amdgcn.alloca : !v
    %gAd0 = amdgcn.alloca : !v
    %gAd1 = amdgcn.alloca : !v
    %gAdst = amdgcn.make_register_range %gAd0, %gAd1 : !v, !v

    // buffer-load B: sx4 rsrc + soffset + voffset -> vx2.
    %rb0 = amdgcn.alloca : !s
    %rb1 = amdgcn.alloca : !s
    %rb2 = amdgcn.alloca : !s
    %rb3 = amdgcn.alloca : !s
    %rsrcB = amdgcn.make_register_range %rb0, %rb1, %rb2, %rb3 : !s, !s, !s, !s
    %Bvoff = amdgcn.alloca : !v
    %Bsoff = amdgcn.alloca : !s
    %Bd0 = amdgcn.alloca : !v
    %Bd1 = amdgcn.alloca : !v
    %Bdst = amdgcn.make_register_range %Bd0, %Bd1 : !v, !v

    // LDS addressing: distinct buffers per access (writes target next-iteration
    // tiles; reads consume the current ones) so memdep does not order the
    // independent reads against the writes.
    %perthread = amdgcn.alloca : !v
    %bufAw = amdgcn.alloc_lds 256 alignment 16
    %bufBw = amdgcn.alloc_lds 256 alignment 16
    %bufAr = amdgcn.alloc_lds 256 alignment 16
    %bufBr = amdgcn.alloc_lds 256 alignment 16
    %offAw = amdgcn.get_lds_offset %bufAw : i32
    %offBw = amdgcn.get_lds_offset %bufBw : i32
    %offAr = amdgcn.get_lds_offset %bufAr : i32
    %offBr = amdgcn.get_lds_offset %bufBr : i32

    // ds_read destinations feeding MFMA.
    %fa0 = amdgcn.alloca : !v
    %fa1 = amdgcn.alloca : !v
    %fragA = amdgcn.make_register_range %fa0, %fa1 : !v, !v
    %fb0 = amdgcn.alloca : !v
    %fb1 = amdgcn.alloca : !v
    %fragB = amdgcn.make_register_range %fb0, %fb1 : !v, !v

    // Independent VALU work.
    %vd0 = amdgcn.alloca : !v
    %vd1 = amdgcn.alloca : !v

    %c0i = arith.constant 0 : i32

    // 1) loads (long latency, mutually independent).
    %grA, %gtA = amdgcn.global_load_dwordx2 dest %gAdst addr %gA offset d(%gAoff) + c(%c0i)
        : outs(!vx2) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %brB, %btB = amdgcn.buffer_load_dwordx2 dest %Bdst addr %rsrcB offset u(%Bsoff) + off_idx(%Bvoff) + c(%c0i) {offen}
        : outs(!vx2) ins(!sx4, !s, !v) mods(i32) -> !amdgcn.read_token<flat>
    // 2) write both tiles to LDS (consume the loads -> VMEM->LGKM edge).
    %wtA = amdgcn.ds_write_b64 data %grA addr %perthread offset c(%offAw)
        : ins(!vx2, !v) mods(i32) -> !amdgcn.write_token<shared>
    %wtB = amdgcn.ds_write_b64 data %brB addr %perthread offset c(%offBw)
        : ins(!vx2, !v) mods(i32) -> !amdgcn.write_token<shared>
    // 3) read fragments back from LDS (LGKM) feeding the MFMA.
    %rdA, %rtA = amdgcn.ds_read_b64 dest %fragA addr %perthread offset c(%offAr)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rdB, %rtB = amdgcn.ds_read_b64 dest %fragB addr %perthread offset c(%offBr)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    // 4) MFMA accumulate chain (XDL), consumes the fragments.
    %m0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc0) ins(%rdA, %rdB, %acc0)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m1 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc1) ins(%rdA, %rdB, %acc1)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    // 5) independent VALU (no dep on the chain -> hoistable into shadows).
    %valu0 = amdgcn.v_mov_b32 outs(%vd0) ins(%c0i) : outs(!v) ins(i32)
    %valu1 = amdgcn.v_mov_b32 outs(%vd1) ins(%c0i) : outs(!v) ins(i32)
    amdgcn.end_kernel
  }
}

// -----

// R1: 8 independent MFMAs + 8 independent ds_read_b64 with no data deps between
// the two queues.  mfma-gap=2 lgkm-gap=2 must interleave them (M d M d ...).
// Without lgkm-gap all 8 ds_reads would front-cluster before any MFMA.

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // Interleaving check: every v_mfma must be followed by a ds_read_b64 (until
  // the ds_reads are exhausted), so no two consecutive v_mfmas appear without a
  // ds_read in between -- assert that pattern for the first 6 MFMAs.
  // INTL-LABEL:   kernel @two_queue_no_deps
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  amdgcn.kernel @two_queue_no_deps {
    // 8 independent accumulator tiles (distinct reg ranges, no shared source).
    %c0_0 = amdgcn.alloca : !v
    %c0_1 = amdgcn.alloca : !v
    %c0_2 = amdgcn.alloca : !v
    %c0_3 = amdgcn.alloca : !v
    %acc0 = amdgcn.make_register_range %c0_0, %c0_1, %c0_2, %c0_3 : !v, !v, !v, !v
    %c1_0 = amdgcn.alloca : !v
    %c1_1 = amdgcn.alloca : !v
    %c1_2 = amdgcn.alloca : !v
    %c1_3 = amdgcn.alloca : !v
    %acc1 = amdgcn.make_register_range %c1_0, %c1_1, %c1_2, %c1_3 : !v, !v, !v, !v
    %c2_0 = amdgcn.alloca : !v
    %c2_1 = amdgcn.alloca : !v
    %c2_2 = amdgcn.alloca : !v
    %c2_3 = amdgcn.alloca : !v
    %acc2 = amdgcn.make_register_range %c2_0, %c2_1, %c2_2, %c2_3 : !v, !v, !v, !v
    %c3_0 = amdgcn.alloca : !v
    %c3_1 = amdgcn.alloca : !v
    %c3_2 = amdgcn.alloca : !v
    %c3_3 = amdgcn.alloca : !v
    %acc3 = amdgcn.make_register_range %c3_0, %c3_1, %c3_2, %c3_3 : !v, !v, !v, !v
    %c4_0 = amdgcn.alloca : !v
    %c4_1 = amdgcn.alloca : !v
    %c4_2 = amdgcn.alloca : !v
    %c4_3 = amdgcn.alloca : !v
    %acc4 = amdgcn.make_register_range %c4_0, %c4_1, %c4_2, %c4_3 : !v, !v, !v, !v
    %c5_0 = amdgcn.alloca : !v
    %c5_1 = amdgcn.alloca : !v
    %c5_2 = amdgcn.alloca : !v
    %c5_3 = amdgcn.alloca : !v
    %acc5 = amdgcn.make_register_range %c5_0, %c5_1, %c5_2, %c5_3 : !v, !v, !v, !v
    %c6_0 = amdgcn.alloca : !v
    %c6_1 = amdgcn.alloca : !v
    %c6_2 = amdgcn.alloca : !v
    %c6_3 = amdgcn.alloca : !v
    %acc6 = amdgcn.make_register_range %c6_0, %c6_1, %c6_2, %c6_3 : !v, !v, !v, !v
    %c7_0 = amdgcn.alloca : !v
    %c7_1 = amdgcn.alloca : !v
    %c7_2 = amdgcn.alloca : !v
    %c7_3 = amdgcn.alloca : !v
    %acc7 = amdgcn.make_register_range %c7_0, %c7_1, %c7_2, %c7_3 : !v, !v, !v, !v

    // 8 independent fragment dest pairs (ds_reads write these; MFMAs read
    // fragBN slots which are distinct -- so MFMAs have NO dep on ds_reads).
    %fa0_0 = amdgcn.alloca : !v
    %fa0_1 = amdgcn.alloca : !v
    %fragA0 = amdgcn.make_register_range %fa0_0, %fa0_1 : !v, !v
    %fb0_0 = amdgcn.alloca : !v
    %fb0_1 = amdgcn.alloca : !v
    %fragB0 = amdgcn.make_register_range %fb0_0, %fb0_1 : !v, !v
    %fa1_0 = amdgcn.alloca : !v
    %fa1_1 = amdgcn.alloca : !v
    %fragA1 = amdgcn.make_register_range %fa1_0, %fa1_1 : !v, !v
    %fb1_0 = amdgcn.alloca : !v
    %fb1_1 = amdgcn.alloca : !v
    %fragB1 = amdgcn.make_register_range %fb1_0, %fb1_1 : !v, !v
    %fa2_0 = amdgcn.alloca : !v
    %fa2_1 = amdgcn.alloca : !v
    %fragA2 = amdgcn.make_register_range %fa2_0, %fa2_1 : !v, !v
    %fb2_0 = amdgcn.alloca : !v
    %fb2_1 = amdgcn.alloca : !v
    %fragB2 = amdgcn.make_register_range %fb2_0, %fb2_1 : !v, !v
    %fa3_0 = amdgcn.alloca : !v
    %fa3_1 = amdgcn.alloca : !v
    %fragA3 = amdgcn.make_register_range %fa3_0, %fa3_1 : !v, !v
    %fb3_0 = amdgcn.alloca : !v
    %fb3_1 = amdgcn.alloca : !v
    %fragB3 = amdgcn.make_register_range %fb3_0, %fb3_1 : !v, !v
    %fa4_0 = amdgcn.alloca : !v
    %fa4_1 = amdgcn.alloca : !v
    %fragA4 = amdgcn.make_register_range %fa4_0, %fa4_1 : !v, !v
    %fb4_0 = amdgcn.alloca : !v
    %fb4_1 = amdgcn.alloca : !v
    %fragB4 = amdgcn.make_register_range %fb4_0, %fb4_1 : !v, !v
    %fa5_0 = amdgcn.alloca : !v
    %fa5_1 = amdgcn.alloca : !v
    %fragA5 = amdgcn.make_register_range %fa5_0, %fa5_1 : !v, !v
    %fb5_0 = amdgcn.alloca : !v
    %fb5_1 = amdgcn.alloca : !v
    %fragB5 = amdgcn.make_register_range %fb5_0, %fb5_1 : !v, !v
    %fa6_0 = amdgcn.alloca : !v
    %fa6_1 = amdgcn.alloca : !v
    %fragA6 = amdgcn.make_register_range %fa6_0, %fa6_1 : !v, !v
    %fb6_0 = amdgcn.alloca : !v
    %fb6_1 = amdgcn.alloca : !v
    %fragB6 = amdgcn.make_register_range %fb6_0, %fb6_1 : !v, !v
    %fa7_0 = amdgcn.alloca : !v
    %fa7_1 = amdgcn.alloca : !v
    %fragA7 = amdgcn.make_register_range %fa7_0, %fa7_1 : !v, !v
    %fb7_0 = amdgcn.alloca : !v
    %fb7_1 = amdgcn.alloca : !v
    %fragB7 = amdgcn.make_register_range %fb7_0, %fb7_1 : !v, !v

    // 8 distinct LDS read addresses (no shared bases -> no dep edges between ds_reads).
    %la0 = amdgcn.alloca : !v
    %la1 = amdgcn.alloca : !v
    %la2 = amdgcn.alloca : !v
    %la3 = amdgcn.alloca : !v
    %la4 = amdgcn.alloca : !v
    %la5 = amdgcn.alloca : !v
    %la6 = amdgcn.alloca : !v
    %la7 = amdgcn.alloca : !v

    %c0i = arith.constant 0 : i32

    // 8 independent ds_read_b64 (write fragAi, no dep on any MFMA below).
    %rd0, %rt0 = amdgcn.ds_read_b64 dest %fragA0 addr %la0 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd1, %rt1 = amdgcn.ds_read_b64 dest %fragA1 addr %la1 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd2, %rt2 = amdgcn.ds_read_b64 dest %fragA2 addr %la2 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd3, %rt3 = amdgcn.ds_read_b64 dest %fragA3 addr %la3 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd4, %rt4 = amdgcn.ds_read_b64 dest %fragA4 addr %la4 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd5, %rt5 = amdgcn.ds_read_b64 dest %fragA5 addr %la5 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd6, %rt6 = amdgcn.ds_read_b64 dest %fragA6 addr %la6 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd7, %rt7 = amdgcn.ds_read_b64 dest %fragA7 addr %la7 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>

    // 8 independent MFMAs (read fragBi -- distinct from fragAi above).
    %m0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc0) ins(%fragB0, %fragB0, %acc0)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m1 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc1) ins(%fragB1, %fragB1, %acc1)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m2 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc2) ins(%fragB2, %fragB2, %acc2)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m3 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc3) ins(%fragB3, %fragB3, %acc3)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m4 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc4) ins(%fragB4, %fragB4, %acc4)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m5 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc5) ins(%fragB5, %fragB5, %acc5)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m6 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc6) ins(%fragB6, %fragB6, %acc6)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m7 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc7) ins(%fragB7, %fragB7, %acc7)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    amdgcn.end_kernel
  }
}

// -----

// R2: 8 ds_read_b64 each feeding exactly one MFMA (RAW dep ds_i -> mfma_i).
// The pairs must interleave (ds0,mfma0,ds1,mfma1,...) not cluster (ds*,mfma*).
// Every ds_read must precede its consuming mfma (data precedence).

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // Interleaving check: every ds_read is immediately followed by the mfma that
  // consumes it (ds0 -> mfma0, ds1 -> mfma1, ...) -- pairs appear in turn.
  // Also verifies data precedence: no mfma appears without its ds_read above.
  // INTL-LABEL:   kernel @fed_ds_mfma
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  amdgcn.kernel @fed_ds_mfma {
    // 8 accumulators (distinct reg ranges).
    %c0_0 = amdgcn.alloca : !v
    %c0_1 = amdgcn.alloca : !v
    %c0_2 = amdgcn.alloca : !v
    %c0_3 = amdgcn.alloca : !v
    %acc0 = amdgcn.make_register_range %c0_0, %c0_1, %c0_2, %c0_3 : !v, !v, !v, !v
    %c1_0 = amdgcn.alloca : !v
    %c1_1 = amdgcn.alloca : !v
    %c1_2 = amdgcn.alloca : !v
    %c1_3 = amdgcn.alloca : !v
    %acc1 = amdgcn.make_register_range %c1_0, %c1_1, %c1_2, %c1_3 : !v, !v, !v, !v
    %c2_0 = amdgcn.alloca : !v
    %c2_1 = amdgcn.alloca : !v
    %c2_2 = amdgcn.alloca : !v
    %c2_3 = amdgcn.alloca : !v
    %acc2 = amdgcn.make_register_range %c2_0, %c2_1, %c2_2, %c2_3 : !v, !v, !v, !v
    %c3_0 = amdgcn.alloca : !v
    %c3_1 = amdgcn.alloca : !v
    %c3_2 = amdgcn.alloca : !v
    %c3_3 = amdgcn.alloca : !v
    %acc3 = amdgcn.make_register_range %c3_0, %c3_1, %c3_2, %c3_3 : !v, !v, !v, !v
    %c4_0 = amdgcn.alloca : !v
    %c4_1 = amdgcn.alloca : !v
    %c4_2 = amdgcn.alloca : !v
    %c4_3 = amdgcn.alloca : !v
    %acc4 = amdgcn.make_register_range %c4_0, %c4_1, %c4_2, %c4_3 : !v, !v, !v, !v
    %c5_0 = amdgcn.alloca : !v
    %c5_1 = amdgcn.alloca : !v
    %c5_2 = amdgcn.alloca : !v
    %c5_3 = amdgcn.alloca : !v
    %acc5 = amdgcn.make_register_range %c5_0, %c5_1, %c5_2, %c5_3 : !v, !v, !v, !v
    %c6_0 = amdgcn.alloca : !v
    %c6_1 = amdgcn.alloca : !v
    %c6_2 = amdgcn.alloca : !v
    %c6_3 = amdgcn.alloca : !v
    %acc6 = amdgcn.make_register_range %c6_0, %c6_1, %c6_2, %c6_3 : !v, !v, !v, !v
    %c7_0 = amdgcn.alloca : !v
    %c7_1 = amdgcn.alloca : !v
    %c7_2 = amdgcn.alloca : !v
    %c7_3 = amdgcn.alloca : !v
    %acc7 = amdgcn.make_register_range %c7_0, %c7_1, %c7_2, %c7_3 : !v, !v, !v, !v

    // 8 fragment dest regs; each ds_read writes fragi, its paired mfma reads it.
    %fa0_0 = amdgcn.alloca : !v
    %fa0_1 = amdgcn.alloca : !v
    %frag0 = amdgcn.make_register_range %fa0_0, %fa0_1 : !v, !v
    %fa1_0 = amdgcn.alloca : !v
    %fa1_1 = amdgcn.alloca : !v
    %frag1 = amdgcn.make_register_range %fa1_0, %fa1_1 : !v, !v
    %fa2_0 = amdgcn.alloca : !v
    %fa2_1 = amdgcn.alloca : !v
    %frag2 = amdgcn.make_register_range %fa2_0, %fa2_1 : !v, !v
    %fa3_0 = amdgcn.alloca : !v
    %fa3_1 = amdgcn.alloca : !v
    %frag3 = amdgcn.make_register_range %fa3_0, %fa3_1 : !v, !v
    %fa4_0 = amdgcn.alloca : !v
    %fa4_1 = amdgcn.alloca : !v
    %frag4 = amdgcn.make_register_range %fa4_0, %fa4_1 : !v, !v
    %fa5_0 = amdgcn.alloca : !v
    %fa5_1 = amdgcn.alloca : !v
    %frag5 = amdgcn.make_register_range %fa5_0, %fa5_1 : !v, !v
    %fa6_0 = amdgcn.alloca : !v
    %fa6_1 = amdgcn.alloca : !v
    %frag6 = amdgcn.make_register_range %fa6_0, %fa6_1 : !v, !v
    %fa7_0 = amdgcn.alloca : !v
    %fa7_1 = amdgcn.alloca : !v
    %frag7 = amdgcn.make_register_range %fa7_0, %fa7_1 : !v, !v

    // 8 LDS read addresses.
    %la0 = amdgcn.alloca : !v
    %la1 = amdgcn.alloca : !v
    %la2 = amdgcn.alloca : !v
    %la3 = amdgcn.alloca : !v
    %la4 = amdgcn.alloca : !v
    %la5 = amdgcn.alloca : !v
    %la6 = amdgcn.alloca : !v
    %la7 = amdgcn.alloca : !v

    %c0i = arith.constant 0 : i32

    // 8 ds_read_b64 each producing a fragment consumed by its paired mfma (RAW).
    %rd0, %rt0 = amdgcn.ds_read_b64 dest %frag0 addr %la0 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd1, %rt1 = amdgcn.ds_read_b64 dest %frag1 addr %la1 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd2, %rt2 = amdgcn.ds_read_b64 dest %frag2 addr %la2 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd3, %rt3 = amdgcn.ds_read_b64 dest %frag3 addr %la3 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd4, %rt4 = amdgcn.ds_read_b64 dest %frag4 addr %la4 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd5, %rt5 = amdgcn.ds_read_b64 dest %frag5 addr %la5 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd6, %rt6 = amdgcn.ds_read_b64 dest %frag6 addr %la6 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd7, %rt7 = amdgcn.ds_read_b64 dest %frag7 addr %la7 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>

    // 8 MFMAs each consuming the ds_read result as A input (RAW: ds_i -> mfma_i).
    %m0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc0) ins(%rd0, %rd0, %acc0)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m1 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc1) ins(%rd1, %rd1, %acc1)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m2 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc2) ins(%rd2, %rd2, %acc2)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m3 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc3) ins(%rd3, %rd3, %acc3)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m4 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc4) ins(%rd4, %rd4, %acc4)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m5 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc5) ins(%rd5, %rd5, %acc5)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m6 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc6) ins(%rd6, %rd6, %acc6)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m7 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc7) ins(%rd7, %rd7, %acc7)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    amdgcn.end_kernel
  }
}

// -----

// R3: 8 MFMAs : 4 ds_read_b64 ratio (2:1), each ds feeding 2 MFMAs.
// Mirrors the real GEMM steady block (64:32 at full scale).
// The 4 ds_reads must be SPREAD across the MFMA sequence -- not front-clustered.
// Actual scheduled pattern: d m d m d m d m m m m m
// (the 4 ds_reads fill the first 8 positions interleaved with 4 MFMAs;
//  the remaining 4 MFMAs trail once all ds_reads are exhausted).

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // Spreading check: the 4 ds_reads are interleaved with the first 4 MFMAs
  // (alternating d,m pattern in the first 8 real ops).  The final 4 MFMAs
  // follow once all ds_reads are consumed.
  // INTL-LABEL:   kernel @ratio_2to1
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  // After all 4 ds_reads, 4 more MFMAs follow (ds_reads exhausted).
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     v_mfma_f32_16x16x16_f16
  amdgcn.kernel @ratio_2to1 {
    // 8 accumulators (4 pairs; pair i consumes fragment i).
    %c0_0 = amdgcn.alloca : !v
    %c0_1 = amdgcn.alloca : !v
    %c0_2 = amdgcn.alloca : !v
    %c0_3 = amdgcn.alloca : !v
    %acc0 = amdgcn.make_register_range %c0_0, %c0_1, %c0_2, %c0_3 : !v, !v, !v, !v
    %c1_0 = amdgcn.alloca : !v
    %c1_1 = amdgcn.alloca : !v
    %c1_2 = amdgcn.alloca : !v
    %c1_3 = amdgcn.alloca : !v
    %acc1 = amdgcn.make_register_range %c1_0, %c1_1, %c1_2, %c1_3 : !v, !v, !v, !v
    %c2_0 = amdgcn.alloca : !v
    %c2_1 = amdgcn.alloca : !v
    %c2_2 = amdgcn.alloca : !v
    %c2_3 = amdgcn.alloca : !v
    %acc2 = amdgcn.make_register_range %c2_0, %c2_1, %c2_2, %c2_3 : !v, !v, !v, !v
    %c3_0 = amdgcn.alloca : !v
    %c3_1 = amdgcn.alloca : !v
    %c3_2 = amdgcn.alloca : !v
    %c3_3 = amdgcn.alloca : !v
    %acc3 = amdgcn.make_register_range %c3_0, %c3_1, %c3_2, %c3_3 : !v, !v, !v, !v
    %c4_0 = amdgcn.alloca : !v
    %c4_1 = amdgcn.alloca : !v
    %c4_2 = amdgcn.alloca : !v
    %c4_3 = amdgcn.alloca : !v
    %acc4 = amdgcn.make_register_range %c4_0, %c4_1, %c4_2, %c4_3 : !v, !v, !v, !v
    %c5_0 = amdgcn.alloca : !v
    %c5_1 = amdgcn.alloca : !v
    %c5_2 = amdgcn.alloca : !v
    %c5_3 = amdgcn.alloca : !v
    %acc5 = amdgcn.make_register_range %c5_0, %c5_1, %c5_2, %c5_3 : !v, !v, !v, !v
    %c6_0 = amdgcn.alloca : !v
    %c6_1 = amdgcn.alloca : !v
    %c6_2 = amdgcn.alloca : !v
    %c6_3 = amdgcn.alloca : !v
    %acc6 = amdgcn.make_register_range %c6_0, %c6_1, %c6_2, %c6_3 : !v, !v, !v, !v
    %c7_0 = amdgcn.alloca : !v
    %c7_1 = amdgcn.alloca : !v
    %c7_2 = amdgcn.alloca : !v
    %c7_3 = amdgcn.alloca : !v
    %acc7 = amdgcn.make_register_range %c7_0, %c7_1, %c7_2, %c7_3 : !v, !v, !v, !v

    // 4 fragment regs; ds_i feeds mfma(2i) and mfma(2i+1).
    %fa0_0 = amdgcn.alloca : !v
    %fa0_1 = amdgcn.alloca : !v
    %frag0 = amdgcn.make_register_range %fa0_0, %fa0_1 : !v, !v
    %fa1_0 = amdgcn.alloca : !v
    %fa1_1 = amdgcn.alloca : !v
    %frag1 = amdgcn.make_register_range %fa1_0, %fa1_1 : !v, !v
    %fa2_0 = amdgcn.alloca : !v
    %fa2_1 = amdgcn.alloca : !v
    %frag2 = amdgcn.make_register_range %fa2_0, %fa2_1 : !v, !v
    %fa3_0 = amdgcn.alloca : !v
    %fa3_1 = amdgcn.alloca : !v
    %frag3 = amdgcn.make_register_range %fa3_0, %fa3_1 : !v, !v

    // 4 LDS read addresses.
    %la0 = amdgcn.alloca : !v
    %la1 = amdgcn.alloca : !v
    %la2 = amdgcn.alloca : !v
    %la3 = amdgcn.alloca : !v

    %c0i = arith.constant 0 : i32

    // 4 ds_read_b64; each result feeds 2 MFMAs (RAW fan-out).
    %rd0, %rt0 = amdgcn.ds_read_b64 dest %frag0 addr %la0 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd1, %rt1 = amdgcn.ds_read_b64 dest %frag1 addr %la1 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd2, %rt2 = amdgcn.ds_read_b64 dest %frag2 addr %la2 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd3, %rt3 = amdgcn.ds_read_b64 dest %frag3 addr %la3 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>

    // 8 MFMAs: pair (m0,m1) consume rd0, (m2,m3) rd1, (m4,m5) rd2, (m6,m7) rd3.
    %m0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc0) ins(%rd0, %rd0, %acc0)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m1 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc1) ins(%rd0, %rd0, %acc1)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m2 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc2) ins(%rd1, %rd1, %acc2)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m3 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc3) ins(%rd1, %rd1, %acc3)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m4 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc4) ins(%rd2, %rd2, %acc4)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m5 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc5) ins(%rd2, %rd2, %acc5)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m6 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc6) ins(%rd3, %rd3, %acc6)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m7 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc7) ins(%rd3, %rd3, %acc7)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    amdgcn.end_kernel
  }
}

// -----

// R4: 6 three-op chains: v_add_u32 (VALU) -> ds_read_b64 (LGKM) -> v_mfma (XDL).
// The address computation (v_add) feeds the LDS address; the fragment feeds MFMA.
// Actual scheduled pattern:
//   v_add x6 (VALU cluster -- no lgkm-gap constraint on VALU), then
//   ds_read / v_mfma x6 interleaved (lgkm-gap=2 spaces the ds_reads).
// The v_adds issue first because they have the deepest critical path to MFMA.

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // Scheduling check:
  //   1. All 6 v_add_u32 (address computation) appear before any ds_read_b64.
  //   2. ds_read_b64 / v_mfma pairs interleave after the adds.
  // INTL-LABEL:   kernel @addr_chain_add_ds_mfma
  //       INTL:     v_add_u32
  //       INTL:     v_add_u32
  //       INTL:     v_add_u32
  //       INTL:     v_add_u32
  //       INTL:     v_add_u32
  //       INTL:     v_add_u32
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  //       INTL:     ds_read_b64
  //       INTL:     v_mfma_f32_16x16x16_f16
  amdgcn.kernel @addr_chain_add_ds_mfma {
    // 6 accumulators.
    %c0_0 = amdgcn.alloca : !v
    %c0_1 = amdgcn.alloca : !v
    %c0_2 = amdgcn.alloca : !v
    %c0_3 = amdgcn.alloca : !v
    %acc0 = amdgcn.make_register_range %c0_0, %c0_1, %c0_2, %c0_3 : !v, !v, !v, !v
    %c1_0 = amdgcn.alloca : !v
    %c1_1 = amdgcn.alloca : !v
    %c1_2 = amdgcn.alloca : !v
    %c1_3 = amdgcn.alloca : !v
    %acc1 = amdgcn.make_register_range %c1_0, %c1_1, %c1_2, %c1_3 : !v, !v, !v, !v
    %c2_0 = amdgcn.alloca : !v
    %c2_1 = amdgcn.alloca : !v
    %c2_2 = amdgcn.alloca : !v
    %c2_3 = amdgcn.alloca : !v
    %acc2 = amdgcn.make_register_range %c2_0, %c2_1, %c2_2, %c2_3 : !v, !v, !v, !v
    %c3_0 = amdgcn.alloca : !v
    %c3_1 = amdgcn.alloca : !v
    %c3_2 = amdgcn.alloca : !v
    %c3_3 = amdgcn.alloca : !v
    %acc3 = amdgcn.make_register_range %c3_0, %c3_1, %c3_2, %c3_3 : !v, !v, !v, !v
    %c4_0 = amdgcn.alloca : !v
    %c4_1 = amdgcn.alloca : !v
    %c4_2 = amdgcn.alloca : !v
    %c4_3 = amdgcn.alloca : !v
    %acc4 = amdgcn.make_register_range %c4_0, %c4_1, %c4_2, %c4_3 : !v, !v, !v, !v
    %c5_0 = amdgcn.alloca : !v
    %c5_1 = amdgcn.alloca : !v
    %c5_2 = amdgcn.alloca : !v
    %c5_3 = amdgcn.alloca : !v
    %acc5 = amdgcn.make_register_range %c5_0, %c5_1, %c5_2, %c5_3 : !v, !v, !v, !v

    // 6 fragment dest regs (ds_read writes each; MFMA reads it).
    %fa0_0 = amdgcn.alloca : !v
    %fa0_1 = amdgcn.alloca : !v
    %frag0 = amdgcn.make_register_range %fa0_0, %fa0_1 : !v, !v
    %fa1_0 = amdgcn.alloca : !v
    %fa1_1 = amdgcn.alloca : !v
    %frag1 = amdgcn.make_register_range %fa1_0, %fa1_1 : !v, !v
    %fa2_0 = amdgcn.alloca : !v
    %fa2_1 = amdgcn.alloca : !v
    %frag2 = amdgcn.make_register_range %fa2_0, %fa2_1 : !v, !v
    %fa3_0 = amdgcn.alloca : !v
    %fa3_1 = amdgcn.alloca : !v
    %frag3 = amdgcn.make_register_range %fa3_0, %fa3_1 : !v, !v
    %fa4_0 = amdgcn.alloca : !v
    %fa4_1 = amdgcn.alloca : !v
    %frag4 = amdgcn.make_register_range %fa4_0, %fa4_1 : !v, !v
    %fa5_0 = amdgcn.alloca : !v
    %fa5_1 = amdgcn.alloca : !v
    %frag5 = amdgcn.make_register_range %fa5_0, %fa5_1 : !v, !v

    // Address computation inputs (base + offset VGPR per chain).
    %base0 = amdgcn.alloca : !v
    %off0  = amdgcn.alloca : !v
    %base1 = amdgcn.alloca : !v
    %off1  = amdgcn.alloca : !v
    %base2 = amdgcn.alloca : !v
    %off2  = amdgcn.alloca : !v
    %base3 = amdgcn.alloca : !v
    %off3  = amdgcn.alloca : !v
    %base4 = amdgcn.alloca : !v
    %off4  = amdgcn.alloca : !v
    %base5 = amdgcn.alloca : !v
    %off5  = amdgcn.alloca : !v

    // v_add_u32 dest regs (hold computed LDS addresses for ds_reads).
    %addr0 = amdgcn.alloca : !v
    %addr1 = amdgcn.alloca : !v
    %addr2 = amdgcn.alloca : !v
    %addr3 = amdgcn.alloca : !v
    %addr4 = amdgcn.alloca : !v
    %addr5 = amdgcn.alloca : !v

    %c0i = arith.constant 0 : i32

    // 6 chains: v_add_u32 (VALU) produces addr -> ds_read_b64 (LGKM) uses addr
    // and produces fragment -> v_mfma_f32 (XDL) uses fragment.
    %a0 = amdgcn.v_add_u32 outs(%addr0) ins(%base0, %off0) : outs(!v) ins(!v, !v)
    %rd0, %rt0 = amdgcn.ds_read_b64 dest %frag0 addr %a0 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc0) ins(%rd0, %rd0, %acc0)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)

    %a1 = amdgcn.v_add_u32 outs(%addr1) ins(%base1, %off1) : outs(!v) ins(!v, !v)
    %rd1, %rt1 = amdgcn.ds_read_b64 dest %frag1 addr %a1 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m1 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc1) ins(%rd1, %rd1, %acc1)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)

    %a2 = amdgcn.v_add_u32 outs(%addr2) ins(%base2, %off2) : outs(!v) ins(!v, !v)
    %rd2, %rt2 = amdgcn.ds_read_b64 dest %frag2 addr %a2 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m2 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc2) ins(%rd2, %rd2, %acc2)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)

    %a3 = amdgcn.v_add_u32 outs(%addr3) ins(%base3, %off3) : outs(!v) ins(!v, !v)
    %rd3, %rt3 = amdgcn.ds_read_b64 dest %frag3 addr %a3 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m3 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc3) ins(%rd3, %rd3, %acc3)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)

    %a4 = amdgcn.v_add_u32 outs(%addr4) ins(%base4, %off4) : outs(!v) ins(!v, !v)
    %rd4, %rt4 = amdgcn.ds_read_b64 dest %frag4 addr %a4 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m4 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc4) ins(%rd4, %rd4, %acc4)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)

    %a5 = amdgcn.v_add_u32 outs(%addr5) ins(%base5, %off5) : outs(!v) ins(!v, !v)
    %rd5, %rt5 = amdgcn.ds_read_b64 dest %frag5 addr %a5 offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m5 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc5) ins(%rd5, %rd5, %acc5)
            : outs(!vx4) ins(!vx2, !vx2, !vx4)
    amdgcn.end_kernel
  }
}

// -----

!v   = !amdgcn.vgpr
!vx4 = !amdgcn.vgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx950> {

  // R6: 16 MFMAs : 8 ds_read_b128 (2:1), CDNA4 v_mfma_f32_16x16x32_f16.
  // Whole-block: 8 tight d,m pairs in head, 8 trailing MFMAs.
  // INTL-LABEL:   kernel @steady_kstep
  //       INTL:     ds_read_b128
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     ds_read_b128
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     ds_read_b128
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     ds_read_b128
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     ds_read_b128
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     ds_read_b128
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     ds_read_b128
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     ds_read_b128
  //       INTL:     v_mfma_f32_16x16x32_f16
  // Trailing tail: 8 MFMAs once ds_reads exhausted.
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     v_mfma_f32_16x16x32_f16
  //       INTL:     v_mfma_f32_16x16x32_f16

  amdgcn.kernel @steady_kstep {
    // 16 accumulator tiles (vx4, one per MFMA).
    %c0_0 = amdgcn.alloca : !v
    %c0_1 = amdgcn.alloca : !v
    %c0_2 = amdgcn.alloca : !v
    %c0_3 = amdgcn.alloca : !v
    %acc0 = amdgcn.make_register_range %c0_0, %c0_1, %c0_2, %c0_3 : !v, !v, !v, !v
    %c1_0 = amdgcn.alloca : !v
    %c1_1 = amdgcn.alloca : !v
    %c1_2 = amdgcn.alloca : !v
    %c1_3 = amdgcn.alloca : !v
    %acc1 = amdgcn.make_register_range %c1_0, %c1_1, %c1_2, %c1_3 : !v, !v, !v, !v
    %c2_0 = amdgcn.alloca : !v
    %c2_1 = amdgcn.alloca : !v
    %c2_2 = amdgcn.alloca : !v
    %c2_3 = amdgcn.alloca : !v
    %acc2 = amdgcn.make_register_range %c2_0, %c2_1, %c2_2, %c2_3 : !v, !v, !v, !v
    %c3_0 = amdgcn.alloca : !v
    %c3_1 = amdgcn.alloca : !v
    %c3_2 = amdgcn.alloca : !v
    %c3_3 = amdgcn.alloca : !v
    %acc3 = amdgcn.make_register_range %c3_0, %c3_1, %c3_2, %c3_3 : !v, !v, !v, !v
    %c4_0 = amdgcn.alloca : !v
    %c4_1 = amdgcn.alloca : !v
    %c4_2 = amdgcn.alloca : !v
    %c4_3 = amdgcn.alloca : !v
    %acc4 = amdgcn.make_register_range %c4_0, %c4_1, %c4_2, %c4_3 : !v, !v, !v, !v
    %c5_0 = amdgcn.alloca : !v
    %c5_1 = amdgcn.alloca : !v
    %c5_2 = amdgcn.alloca : !v
    %c5_3 = amdgcn.alloca : !v
    %acc5 = amdgcn.make_register_range %c5_0, %c5_1, %c5_2, %c5_3 : !v, !v, !v, !v
    %c6_0 = amdgcn.alloca : !v
    %c6_1 = amdgcn.alloca : !v
    %c6_2 = amdgcn.alloca : !v
    %c6_3 = amdgcn.alloca : !v
    %acc6 = amdgcn.make_register_range %c6_0, %c6_1, %c6_2, %c6_3 : !v, !v, !v, !v
    %c7_0 = amdgcn.alloca : !v
    %c7_1 = amdgcn.alloca : !v
    %c7_2 = amdgcn.alloca : !v
    %c7_3 = amdgcn.alloca : !v
    %acc7 = amdgcn.make_register_range %c7_0, %c7_1, %c7_2, %c7_3 : !v, !v, !v, !v
    %c8_0 = amdgcn.alloca : !v
    %c8_1 = amdgcn.alloca : !v
    %c8_2 = amdgcn.alloca : !v
    %c8_3 = amdgcn.alloca : !v
    %acc8 = amdgcn.make_register_range %c8_0, %c8_1, %c8_2, %c8_3 : !v, !v, !v, !v
    %c9_0 = amdgcn.alloca : !v
    %c9_1 = amdgcn.alloca : !v
    %c9_2 = amdgcn.alloca : !v
    %c9_3 = amdgcn.alloca : !v
    %acc9 = amdgcn.make_register_range %c9_0, %c9_1, %c9_2, %c9_3 : !v, !v, !v, !v
    %ca_0 = amdgcn.alloca : !v
    %ca_1 = amdgcn.alloca : !v
    %ca_2 = amdgcn.alloca : !v
    %ca_3 = amdgcn.alloca : !v
    %acca = amdgcn.make_register_range %ca_0, %ca_1, %ca_2, %ca_3 : !v, !v, !v, !v
    %cb_0 = amdgcn.alloca : !v
    %cb_1 = amdgcn.alloca : !v
    %cb_2 = amdgcn.alloca : !v
    %cb_3 = amdgcn.alloca : !v
    %accb = amdgcn.make_register_range %cb_0, %cb_1, %cb_2, %cb_3 : !v, !v, !v, !v
    %cc_0 = amdgcn.alloca : !v
    %cc_1 = amdgcn.alloca : !v
    %cc_2 = amdgcn.alloca : !v
    %cc_3 = amdgcn.alloca : !v
    %accc = amdgcn.make_register_range %cc_0, %cc_1, %cc_2, %cc_3 : !v, !v, !v, !v
    %cd_0 = amdgcn.alloca : !v
    %cd_1 = amdgcn.alloca : !v
    %cd_2 = amdgcn.alloca : !v
    %cd_3 = amdgcn.alloca : !v
    %accd = amdgcn.make_register_range %cd_0, %cd_1, %cd_2, %cd_3 : !v, !v, !v, !v
    %ce_0 = amdgcn.alloca : !v
    %ce_1 = amdgcn.alloca : !v
    %ce_2 = amdgcn.alloca : !v
    %ce_3 = amdgcn.alloca : !v
    %acce = amdgcn.make_register_range %ce_0, %ce_1, %ce_2, %ce_3 : !v, !v, !v, !v
    %cf_0 = amdgcn.alloca : !v
    %cf_1 = amdgcn.alloca : !v
    %cf_2 = amdgcn.alloca : !v
    %cf_3 = amdgcn.alloca : !v
    %accf = amdgcn.make_register_range %cf_0, %cf_1, %cf_2, %cf_3 : !v, !v, !v, !v

    // 8 fragment dest regs (vx4 for ds_read_b128, K=32 doubled-K operand size).
    %f0_0 = amdgcn.alloca : !v
    %f0_1 = amdgcn.alloca : !v
    %f0_2 = amdgcn.alloca : !v
    %f0_3 = amdgcn.alloca : !v
    %frag0 = amdgcn.make_register_range %f0_0, %f0_1, %f0_2, %f0_3 : !v, !v, !v, !v
    %f1_0 = amdgcn.alloca : !v
    %f1_1 = amdgcn.alloca : !v
    %f1_2 = amdgcn.alloca : !v
    %f1_3 = amdgcn.alloca : !v
    %frag1 = amdgcn.make_register_range %f1_0, %f1_1, %f1_2, %f1_3 : !v, !v, !v, !v
    %f2_0 = amdgcn.alloca : !v
    %f2_1 = amdgcn.alloca : !v
    %f2_2 = amdgcn.alloca : !v
    %f2_3 = amdgcn.alloca : !v
    %frag2 = amdgcn.make_register_range %f2_0, %f2_1, %f2_2, %f2_3 : !v, !v, !v, !v
    %f3_0 = amdgcn.alloca : !v
    %f3_1 = amdgcn.alloca : !v
    %f3_2 = amdgcn.alloca : !v
    %f3_3 = amdgcn.alloca : !v
    %frag3 = amdgcn.make_register_range %f3_0, %f3_1, %f3_2, %f3_3 : !v, !v, !v, !v
    %f4_0 = amdgcn.alloca : !v
    %f4_1 = amdgcn.alloca : !v
    %f4_2 = amdgcn.alloca : !v
    %f4_3 = amdgcn.alloca : !v
    %frag4 = amdgcn.make_register_range %f4_0, %f4_1, %f4_2, %f4_3 : !v, !v, !v, !v
    %f5_0 = amdgcn.alloca : !v
    %f5_1 = amdgcn.alloca : !v
    %f5_2 = amdgcn.alloca : !v
    %f5_3 = amdgcn.alloca : !v
    %frag5 = amdgcn.make_register_range %f5_0, %f5_1, %f5_2, %f5_3 : !v, !v, !v, !v
    %f6_0 = amdgcn.alloca : !v
    %f6_1 = amdgcn.alloca : !v
    %f6_2 = amdgcn.alloca : !v
    %f6_3 = amdgcn.alloca : !v
    %frag6 = amdgcn.make_register_range %f6_0, %f6_1, %f6_2, %f6_3 : !v, !v, !v, !v
    %f7_0 = amdgcn.alloca : !v
    %f7_1 = amdgcn.alloca : !v
    %f7_2 = amdgcn.alloca : !v
    %f7_3 = amdgcn.alloca : !v
    %frag7 = amdgcn.make_register_range %f7_0, %f7_1, %f7_2, %f7_3 : !v, !v, !v, !v

    // 8 LDS read addresses.
    %la0 = amdgcn.alloca : !v
    %la1 = amdgcn.alloca : !v
    %la2 = amdgcn.alloca : !v
    %la3 = amdgcn.alloca : !v
    %la4 = amdgcn.alloca : !v
    %la5 = amdgcn.alloca : !v
    %la6 = amdgcn.alloca : !v
    %la7 = amdgcn.alloca : !v

    %c0i = arith.constant 0 : i32

    // 8 ds_read_b128; each result feeds 2 MFMAs (RAW fan-out 2x).
    %rd0, %rt0 = amdgcn.ds_read_b128 dest %frag0 addr %la0 offset c(%c0i)
        : outs(!vx4) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd1, %rt1 = amdgcn.ds_read_b128 dest %frag1 addr %la1 offset c(%c0i)
        : outs(!vx4) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd2, %rt2 = amdgcn.ds_read_b128 dest %frag2 addr %la2 offset c(%c0i)
        : outs(!vx4) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd3, %rt3 = amdgcn.ds_read_b128 dest %frag3 addr %la3 offset c(%c0i)
        : outs(!vx4) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd4, %rt4 = amdgcn.ds_read_b128 dest %frag4 addr %la4 offset c(%c0i)
        : outs(!vx4) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd5, %rt5 = amdgcn.ds_read_b128 dest %frag5 addr %la5 offset c(%c0i)
        : outs(!vx4) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd6, %rt6 = amdgcn.ds_read_b128 dest %frag6 addr %la6 offset c(%c0i)
        : outs(!vx4) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %rd7, %rt7 = amdgcn.ds_read_b128 dest %frag7 addr %la7 offset c(%c0i)
        : outs(!vx4) ins(!v) mods(i32) -> !amdgcn.read_token<shared>

    // 16 MFMAs: pair (m0,m1) consumes rd0, (m2,m3) rd1, ..., (m14,m15) rd7.
    %m0 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc0) ins(%rd0, %rd0, %acc0)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m1 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc1) ins(%rd0, %rd0, %acc1)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m2 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc2) ins(%rd1, %rd1, %acc2)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m3 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc3) ins(%rd1, %rd1, %acc3)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m4 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc4) ins(%rd2, %rd2, %acc4)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m5 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc5) ins(%rd2, %rd2, %acc5)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m6 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc6) ins(%rd3, %rd3, %acc6)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m7 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc7) ins(%rd3, %rd3, %acc7)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m8 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc8) ins(%rd4, %rd4, %acc8)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %m9 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acc9) ins(%rd4, %rd4, %acc9)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %ma = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acca) ins(%rd5, %rd5, %acca)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %mb = amdgcn.v_mfma_f32_16x16x32_f16 outs(%accb) ins(%rd5, %rd5, %accb)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %mc = amdgcn.v_mfma_f32_16x16x32_f16 outs(%accc) ins(%rd6, %rd6, %accc)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %md = amdgcn.v_mfma_f32_16x16x32_f16 outs(%accd) ins(%rd6, %rd6, %accd)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %me = amdgcn.v_mfma_f32_16x16x32_f16 outs(%acce) ins(%rd7, %rd7, %acce)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    %mf = amdgcn.v_mfma_f32_16x16x32_f16 outs(%accf) ins(%rd7, %rd7, %accf)
            : outs(!vx4) ins(!vx4, !vx4, !vx4)
    amdgcn.end_kernel
  }
}
