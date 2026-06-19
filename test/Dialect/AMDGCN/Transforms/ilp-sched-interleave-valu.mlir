// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=2 mfma-gap=2 lgkm-gap=2 ilp-time-limit-ms=2000})))" | FileCheck %s

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
  // CHECK-LABEL:   kernel @addr_chain_add_ds_mfma
  //       CHECK:     v_add_u32
  //       CHECK:     v_add_u32
  //       CHECK:     v_add_u32
  //       CHECK:     v_add_u32
  //       CHECK:     v_add_u32
  //       CHECK:     v_add_u32
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
  //       CHECK:     ds_read_b64
  //       CHECK:     v_mfma_f32_16x16x16_f16
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
