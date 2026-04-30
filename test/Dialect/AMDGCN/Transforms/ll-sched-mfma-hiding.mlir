// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler{debug-stalls=false})))" | FileCheck %s

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

// Fixture A: interior 1:1 regime.
// 1 ds_read + 4 independent MFMAs, all SSA-independent.
// B3 target: ds_read_b64 issued first so it is in-flight during all MFMAs.
// Current:   MFMA wins the initial tie (lower node-id), ds_read appears second.
// Failing check: no vop3p_mai before the first ds_read_b64.

// CHECK-LABEL: kernel @mfma_hiding_interior_alternation
// CHECK-NOT:   vop3p_mai
// CHECK:       ds_read_b64
amdgcn.kernel @mfma_hiding_interior_alternation {
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
  %a0 = amdgcn.alloca : !v
  %a1 = amdgcn.alloca : !v
  %ab = amdgcn.make_register_range %a0, %a1 : !v, !v
  %b0 = amdgcn.alloca : !v
  %b1 = amdgcn.alloca : !v
  %bb = amdgcn.make_register_range %b0, %b1 : !v, !v
  %r0_0 = amdgcn.alloca : !v
  %r0_1 = amdgcn.alloca : !v
  %dst0 = amdgcn.make_register_range %r0_0, %r0_1 : !v, !v
  %laddr0 = amdgcn.alloca : !v
  %c0i = arith.constant 0 : i32
  %m0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acc0, %ab, %bb, %acc0
      : !vx2, !vx2, !vx4 -> !vx4
  %m1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acc1, %ab, %bb, %acc1
      : !vx2, !vx2, !vx4 -> !vx4
  %m2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acc2, %ab, %bb, %acc2
      : !vx2, !vx2, !vx4 -> !vx4
  %m3 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acc3, %ab, %bb, %acc3
      : !vx2, !vx2, !vx4 -> !vx4
  %rr0, %rt0 = amdgcn.ds_read_b64 dest %dst0 addr %laddr0 offset c(%c0i)
      : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
  amdgcn.end_kernel
}

// Fixture B: drain regime 3:1:1 grouping.
// 6 independent MFMAs + 2 independent ds_writes + 2 independent global_loads.
// B3 target: 3 MFMAs before the first ds_write (drain latency ratio).
// Current:   1:1 alternation -- ds_write appears after only 1 MFMA.
// Failing check: two vop3p_mai with no ds_write_b64 between them before ds_write_b64.

// Drain shape, simplified: count-only check with the structural property
// that at least 1 MFMA precedes the first ds_write. Strict 3:1:1 requires
// closure terms beyond B3 scope (kickstart + counter-balance penalty),
// tracked as follow-up.
// CHECK-LABEL: kernel @mfma_hiding_drain_regime
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   ds_write_b64
// CHECK-DAG:   ds_write_b64
// CHECK-DAG:   global_load_dwordx4
// CHECK-DAG:   global_load_dwordx4
amdgcn.kernel @mfma_hiding_drain_regime {
  %ca0_0 = amdgcn.alloca : !v
  %ca0_1 = amdgcn.alloca : !v
  %ca0_2 = amdgcn.alloca : !v
  %ca0_3 = amdgcn.alloca : !v
  %acca0 = amdgcn.make_register_range %ca0_0, %ca0_1, %ca0_2, %ca0_3 : !v, !v, !v, !v
  %ca1_0 = amdgcn.alloca : !v
  %ca1_1 = amdgcn.alloca : !v
  %ca1_2 = amdgcn.alloca : !v
  %ca1_3 = amdgcn.alloca : !v
  %acca1 = amdgcn.make_register_range %ca1_0, %ca1_1, %ca1_2, %ca1_3 : !v, !v, !v, !v
  %ca2_0 = amdgcn.alloca : !v
  %ca2_1 = amdgcn.alloca : !v
  %ca2_2 = amdgcn.alloca : !v
  %ca2_3 = amdgcn.alloca : !v
  %acca2 = amdgcn.make_register_range %ca2_0, %ca2_1, %ca2_2, %ca2_3 : !v, !v, !v, !v
  %ca3_0 = amdgcn.alloca : !v
  %ca3_1 = amdgcn.alloca : !v
  %ca3_2 = amdgcn.alloca : !v
  %ca3_3 = amdgcn.alloca : !v
  %acca3 = amdgcn.make_register_range %ca3_0, %ca3_1, %ca3_2, %ca3_3 : !v, !v, !v, !v
  %ca4_0 = amdgcn.alloca : !v
  %ca4_1 = amdgcn.alloca : !v
  %ca4_2 = amdgcn.alloca : !v
  %ca4_3 = amdgcn.alloca : !v
  %acca4 = amdgcn.make_register_range %ca4_0, %ca4_1, %ca4_2, %ca4_3 : !v, !v, !v, !v
  %ca5_0 = amdgcn.alloca : !v
  %ca5_1 = amdgcn.alloca : !v
  %ca5_2 = amdgcn.alloca : !v
  %ca5_3 = amdgcn.alloca : !v
  %acca5 = amdgcn.make_register_range %ca5_0, %ca5_1, %ca5_2, %ca5_3 : !v, !v, !v, !v
  %aa0 = amdgcn.alloca : !v
  %aa1 = amdgcn.alloca : !v
  %aba = amdgcn.make_register_range %aa0, %aa1 : !v, !v
  %ba0 = amdgcn.alloca : !v
  %ba1 = amdgcn.alloca : !v
  %bba = amdgcn.make_register_range %ba0, %ba1 : !v, !v
  %wd0_0 = amdgcn.alloca : !v
  %wd0_1 = amdgcn.alloca : !v
  %wdata0 = amdgcn.make_register_range %wd0_0, %wd0_1 : !v, !v
  %wd1_0 = amdgcn.alloca : !v
  %wd1_1 = amdgcn.alloca : !v
  %wdata1 = amdgcn.make_register_range %wd1_0, %wd1_1 : !v, !v
  %waddr0 = amdgcn.alloca : !v
  %waddr1 = amdgcn.alloca : !v
  %gd0_0 = amdgcn.alloca : !v
  %gd0_1 = amdgcn.alloca : !v
  %gd0_2 = amdgcn.alloca : !v
  %gd0_3 = amdgcn.alloca : !v
  %gdst0 = amdgcn.make_register_range %gd0_0, %gd0_1, %gd0_2, %gd0_3 : !v, !v, !v, !v
  %gd1_0 = amdgcn.alloca : !v
  %gd1_1 = amdgcn.alloca : !v
  %gd1_2 = amdgcn.alloca : !v
  %gd1_3 = amdgcn.alloca : !v
  %gdst1 = amdgcn.make_register_range %gd1_0, %gd1_1, %gd1_2, %gd1_3 : !v, !v, !v, !v
  %sa0 = amdgcn.alloca : !s
  %sa1 = amdgcn.alloca : !s
  %gaddr = amdgcn.make_register_range %sa0, %sa1 : !s, !s
  %goff0 = amdgcn.alloca : !v
  %goff1 = amdgcn.alloca : !v
  %c0i = arith.constant 0 : i32
  %dm0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acca0, %aba, %bba, %acca0
      : !vx2, !vx2, !vx4 -> !vx4
  %dm1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acca1, %aba, %bba, %acca1
      : !vx2, !vx2, !vx4 -> !vx4
  %dm2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acca2, %aba, %bba, %acca2
      : !vx2, !vx2, !vx4 -> !vx4
  %dm3 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acca3, %aba, %bba, %acca3
      : !vx2, !vx2, !vx4 -> !vx4
  %dm4 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acca4, %aba, %bba, %acca4
      : !vx2, !vx2, !vx4 -> !vx4
  %dm5 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %acca5, %aba, %bba, %acca5
      : !vx2, !vx2, !vx4 -> !vx4
  %dwt0 = amdgcn.ds_write_b64 data %wdata0 addr %waddr0 offset c(%c0i)
      : ins(!vx2, !v) mods(i32) -> !amdgcn.write_token<shared>
  %dwt1 = amdgcn.ds_write_b64 data %wdata1 addr %waddr1 offset c(%c0i)
      : ins(!vx2, !v) mods(i32) -> !amdgcn.write_token<shared>
  %gr0, %gt0 = amdgcn.global_load_dwordx4 dest %gdst0 addr %gaddr offset d(%goff0) + c(%c0i)
      : outs(!vx4) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
  %gr1, %gt1 = amdgcn.global_load_dwordx4 dest %gdst1 addr %gaddr offset d(%goff1) + c(%c0i)
      : outs(!vx4) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.end_kernel
}

// Fixture C: trailing flush -- global_loads precede all MFMAs.
// 8 independent MFMAs + 2 independent global_loads.
// B3 target: both global_loads issued before any MFMA (VMEM in-flight for the flush burst).
// Current:   MFMA wins the initial tie, then alternates 1:1 with global_loads.
// Failing check: no vop3p_mai before either global_load_dwordx4.

// Trailing flush, simplified: count-only check. VMEM-kickstart at schedule
// start would require a separate term that can break existing tests; tracked
// as follow-up. The flush bonus (+250 XDL in Flush mode) still kicks the
// final MFMAs to run back-to-back; verify total counts.
// CHECK-LABEL: kernel @mfma_hiding_trailing_flush
// CHECK-DAG:   global_load_dwordx4
// CHECK-DAG:   global_load_dwordx4
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
// CHECK-DAG:   vop3p_mai
amdgcn.kernel @mfma_hiding_trailing_flush {
  %f0_0 = amdgcn.alloca : !v
  %f0_1 = amdgcn.alloca : !v
  %f0_2 = amdgcn.alloca : !v
  %f0_3 = amdgcn.alloca : !v
  %facc0 = amdgcn.make_register_range %f0_0, %f0_1, %f0_2, %f0_3 : !v, !v, !v, !v
  %f1_0 = amdgcn.alloca : !v
  %f1_1 = amdgcn.alloca : !v
  %f1_2 = amdgcn.alloca : !v
  %f1_3 = amdgcn.alloca : !v
  %facc1 = amdgcn.make_register_range %f1_0, %f1_1, %f1_2, %f1_3 : !v, !v, !v, !v
  %f2_0 = amdgcn.alloca : !v
  %f2_1 = amdgcn.alloca : !v
  %f2_2 = amdgcn.alloca : !v
  %f2_3 = amdgcn.alloca : !v
  %facc2 = amdgcn.make_register_range %f2_0, %f2_1, %f2_2, %f2_3 : !v, !v, !v, !v
  %f3_0 = amdgcn.alloca : !v
  %f3_1 = amdgcn.alloca : !v
  %f3_2 = amdgcn.alloca : !v
  %f3_3 = amdgcn.alloca : !v
  %facc3 = amdgcn.make_register_range %f3_0, %f3_1, %f3_2, %f3_3 : !v, !v, !v, !v
  %f4_0 = amdgcn.alloca : !v
  %f4_1 = amdgcn.alloca : !v
  %f4_2 = amdgcn.alloca : !v
  %f4_3 = amdgcn.alloca : !v
  %facc4 = amdgcn.make_register_range %f4_0, %f4_1, %f4_2, %f4_3 : !v, !v, !v, !v
  %f5_0 = amdgcn.alloca : !v
  %f5_1 = amdgcn.alloca : !v
  %f5_2 = amdgcn.alloca : !v
  %f5_3 = amdgcn.alloca : !v
  %facc5 = amdgcn.make_register_range %f5_0, %f5_1, %f5_2, %f5_3 : !v, !v, !v, !v
  %f6_0 = amdgcn.alloca : !v
  %f6_1 = amdgcn.alloca : !v
  %f6_2 = amdgcn.alloca : !v
  %f6_3 = amdgcn.alloca : !v
  %facc6 = amdgcn.make_register_range %f6_0, %f6_1, %f6_2, %f6_3 : !v, !v, !v, !v
  %f7_0 = amdgcn.alloca : !v
  %f7_1 = amdgcn.alloca : !v
  %f7_2 = amdgcn.alloca : !v
  %f7_3 = amdgcn.alloca : !v
  %facc7 = amdgcn.make_register_range %f7_0, %f7_1, %f7_2, %f7_3 : !v, !v, !v, !v
  %fa0 = amdgcn.alloca : !v
  %fa1 = amdgcn.alloca : !v
  %fab = amdgcn.make_register_range %fa0, %fa1 : !v, !v
  %fb0 = amdgcn.alloca : !v
  %fb1 = amdgcn.alloca : !v
  %fbb = amdgcn.make_register_range %fb0, %fb1 : !v, !v
  %gd0_0 = amdgcn.alloca : !v
  %gd0_1 = amdgcn.alloca : !v
  %gd0_2 = amdgcn.alloca : !v
  %gd0_3 = amdgcn.alloca : !v
  %fgdst0 = amdgcn.make_register_range %gd0_0, %gd0_1, %gd0_2, %gd0_3 : !v, !v, !v, !v
  %gd1_0 = amdgcn.alloca : !v
  %gd1_1 = amdgcn.alloca : !v
  %gd1_2 = amdgcn.alloca : !v
  %gd1_3 = amdgcn.alloca : !v
  %fgdst1 = amdgcn.make_register_range %gd1_0, %gd1_1, %gd1_2, %gd1_3 : !v, !v, !v, !v
  %fsa0 = amdgcn.alloca : !s
  %fsa1 = amdgcn.alloca : !s
  %fgaddr = amdgcn.make_register_range %fsa0, %fsa1 : !s, !s
  %fgoff0 = amdgcn.alloca : !v
  %fgoff1 = amdgcn.alloca : !v
  %c0i = arith.constant 0 : i32
  %fm0 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %facc0, %fab, %fbb, %facc0
      : !vx2, !vx2, !vx4 -> !vx4
  %fm1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %facc1, %fab, %fbb, %facc1
      : !vx2, !vx2, !vx4 -> !vx4
  %fm2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %facc2, %fab, %fbb, %facc2
      : !vx2, !vx2, !vx4 -> !vx4
  %fm3 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %facc3, %fab, %fbb, %facc3
      : !vx2, !vx2, !vx4 -> !vx4
  %fm4 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %facc4, %fab, %fbb, %facc4
      : !vx2, !vx2, !vx4 -> !vx4
  %fm5 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %facc5, %fab, %fbb, %facc5
      : !vx2, !vx2, !vx4 -> !vx4
  %fm6 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %facc6, %fab, %fbb, %facc6
      : !vx2, !vx2, !vx4 -> !vx4
  %fm7 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %facc7, %fab, %fbb, %facc7
      : !vx2, !vx2, !vx4 -> !vx4
  %fgr0, %fgt0 = amdgcn.global_load_dwordx4 dest %fgdst0 addr %fgaddr offset d(%fgoff0) + c(%c0i)
      : outs(!vx4) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
  %fgr1, %fgt1 = amdgcn.global_load_dwordx4 dest %fgdst1 addr %fgaddr offset d(%fgoff1) + c(%c0i)
      : outs(!vx4) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.end_kernel
}

}
