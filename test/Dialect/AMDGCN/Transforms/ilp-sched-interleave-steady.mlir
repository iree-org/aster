// RUN: aster-opt %s \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=2 mfma-gap=2 lgkm-gap=2 ilp-time-limit-ms=2000})))" \
// RUN: | FileCheck %s --check-prefix=WHOLE
// RUN: aster-opt %s \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=2 mfma-gap=2 lgkm-gap=2 ilp-window-mfmas=4 ilp-time-limit-ms=2000})))" \
// RUN: | FileCheck %s --check-prefix=WIN4

!v   = !amdgcn.vgpr
!vx4 = !amdgcn.vgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx950> {

  // R6: 16 MFMAs : 8 ds_read_b128 (2:1), CDNA4 v_mfma_f32_16x16x32_f16.
  // Whole-block: 8 tight d,m pairs in head, 8 trailing MFMAs.
  // WHOLE-LABEL:   kernel @steady_kstep
  //       WHOLE:     ds_read_b128
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     ds_read_b128
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     ds_read_b128
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     ds_read_b128
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     ds_read_b128
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     ds_read_b128
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     ds_read_b128
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     ds_read_b128
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  // Trailing tail: 8 MFMAs once ds_reads exhausted.
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     v_mfma_f32_16x16x32_f16
  //       WHOLE:     v_mfma_f32_16x16x32_f16

  // Windowed (4 MFMA windows): each window sees 2 ds_reads then 4 MFMAs (max run = 4).
  // WIN4-LABEL:   kernel @steady_kstep
  //       WIN4:     ds_read_b128
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     ds_read_b128
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     ds_read_b128
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     ds_read_b128
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     ds_read_b128
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     ds_read_b128
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     ds_read_b128
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     ds_read_b128
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     v_mfma_f32_16x16x32_f16
  //       WIN4:     v_mfma_f32_16x16x32_f16
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
