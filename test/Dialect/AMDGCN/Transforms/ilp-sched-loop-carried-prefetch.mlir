// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=2 mfma-gap=2 lgkm-gap=2 ilp-time-limit-ms=2000})))" --split-input-file | FileCheck %s

// A loop-carried prefetch load is a VMEM op whose result has no in-block
// consumer: it only flows across the loop back edge as an iter-arg consumed
// next iteration. The removed windowing path assigned such consumer-less ops to
// the last window, clustering every prefetch load at the iteration tail; at the
// back edge the whole cluster was in flight at once and the count-based wait
// could not drain it to vmcnt(0) before the next iteration's loads reused the
// same registers, an illegal access on a corrupted load address. The ILP now
// always solves the block whole, spacing the load across the MFMAs with vmemGap:
// the consumer-less load interleaves near the start, not after the first MFMAs.

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // The loop-carried prefetch load interleaves before the 4th MFMA (whole-block
  // spread), rather than clustering after the first MFMA window.
  // CHECK-LABEL: kernel @loop_carried_prefetch
  // CHECK:         ds_read_b64
  // CHECK:         global_load_dwordx2
  // CHECK:         v_mfma_f32_16x16x16_f16
  // CHECK:         v_mfma_f32_16x16x16_f16
  // CHECK:         v_mfma_f32_16x16x16_f16
  // CHECK:         v_mfma_f32_16x16x16_f16
  amdgcn.kernel @loop_carried_prefetch {
    %gd0 = amdgcn.alloca : !v
    %gd1 = amdgcn.alloca : !v
    %gdst = amdgcn.make_register_range %gd0, %gd1 : !v, !v
    %sa0 = amdgcn.alloca : !s
    %sa1 = amdgcn.alloca : !s
    %gaddr = amdgcn.make_register_range %sa0, %sa1 : !s, !s
    %goff = amdgcn.alloca : !v
    %c0i = arith.constant 0 : i32

    %fa0 = amdgcn.alloca : !v
    %fa1 = amdgcn.alloca : !v
    %frag = amdgcn.make_register_range %fa0, %fa1 : !v, !v
    %buf = amdgcn.alloc_lds 256 alignment 16
    %off = amdgcn.get_lds_offset %buf : i32
    %pt = amdgcn.alloca : !v

    %c0 = amdgcn.alloca : !v
    %c1 = amdgcn.alloca : !v
    %c2 = amdgcn.alloca : !v
    %c3 = amdgcn.alloca : !v
    %acc = amdgcn.make_register_range %c0, %c1, %c2, %c3 : !v, !v, !v, !v

    // Loop-carried prefetch load: %lr has no in-block consumer (only the token
    // and result would flow across a back edge in the full kernel).
    %lr, %ltok = amdgcn.global_load_dwordx2 dest %gdst addr %gaddr offset d(%goff) + c(%c0i)
        : outs(!vx2) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
    %rd, %rt = amdgcn.ds_read_b64 dest %frag addr %pt offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %m0 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %acc)
        : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m1 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %m0)
        : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m2 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %m1)
        : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m3 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %m2)
        : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m4 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %m3)
        : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m5 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %m4)
        : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m6 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %m5)
        : outs(!vx4) ins(!vx2, !vx2, !vx4)
    %m7 = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%rd, %rd, %m6)
        : outs(!vx4) ins(!vx2, !vx2, !vx4)
    amdgcn.end_kernel
  }
}
