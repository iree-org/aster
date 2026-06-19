// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=2 mfma-gap=2 lgkm-gap=2 ilp-time-limit-ms=2000})))" | FileCheck %s

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
  // CHECK-LABEL:   kernel @ratio_2to1
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  // After all 4 ds_reads, 4 more MFMAs follow (ds_reads exhausted).
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     v_mfma_f32_16x16x16_f16
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
