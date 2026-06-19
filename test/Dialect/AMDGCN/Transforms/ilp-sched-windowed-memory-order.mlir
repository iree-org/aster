// RUN: aster-opt %s \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=2 mfma-gap=2 lgkm-gap=2 ilp-window-mfmas=4 ilp-time-limit-ms=2000})))" \
// RUN: | FileCheck %s

// Freeze invariant: windowed solve keeps VMEM before LGKM when that is program order.
// global_load (VMEM, no in-block consumer) precedes ds_read (LGKM) in source order.
// Without the freeze edge, the windowed solver sinks global_load to the last window
// (no consumer -> sink) while ds_read goes to window 0 (feeds MFMAs 0-3), inverting
// their relative order.  The freeze edge global_load -> ds_read pulls global_load into
// window 0 and pins it before ds_read.

// CHECK-LABEL:   kernel @windowed_mem_order_freeze
//       CHECK:     global_load_dwordx2
//       CHECK:     ds_read_b64

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr<[? + 2]>

amdgcn.module @test target = #amdgcn.target<gfx942> {
  amdgcn.kernel @windowed_mem_order_freeze {
    // Prefetch load destination (VMEM result has no in-block use).
    %gd0 = amdgcn.alloca : !v
    %gd1 = amdgcn.alloca : !v
    %gdst = amdgcn.make_register_range %gd0, %gd1 : !v, !v

    // SGPR base and VGPR offset for the global_load.
    %sa0 = amdgcn.alloca : !s
    %sa1 = amdgcn.alloca : !s
    %gaddr = amdgcn.make_register_range %sa0, %sa1 : !s, !s
    %goff = amdgcn.alloca : !v

    // LDS read destination and address.
    %fa0 = amdgcn.alloca : !v
    %fa1 = amdgcn.alloca : !v
    %frag = amdgcn.make_register_range %fa0, %fa1 : !v, !v
    %laddr = amdgcn.alloca : !v

    // Accumulator for 8 chained MFMAs.
    %c0 = amdgcn.alloca : !v
    %c1 = amdgcn.alloca : !v
    %c2 = amdgcn.alloca : !v
    %c3 = amdgcn.alloca : !v
    %acc = amdgcn.make_register_range %c0, %c1, %c2, %c3 : !v, !v, !v, !v

    %c0i = arith.constant 0 : i32

    // Program order: VMEM first.  No in-block consumer -> sink window without freeze.
    %lr, %ltok = amdgcn.global_load_dwordx2 dest %gdst addr %gaddr offset d(%goff) + c(%c0i)
        : outs(!vx2) ins(!sx2, !v) mods(i32) -> !amdgcn.read_token<flat>

    // Program order: LGKM second.  Feeds all 8 MFMAs -> window 0 without freeze.
    %rd, %rt = amdgcn.ds_read_b64 dest %frag addr %laddr offset c(%c0i)
        : outs(!vx2) ins(!v) mods(i32) -> !amdgcn.read_token<shared>

    // 8 chained MFMAs: xdlCount=8, ilp-window-mfmas=4 -> windows 0 and 1.
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
