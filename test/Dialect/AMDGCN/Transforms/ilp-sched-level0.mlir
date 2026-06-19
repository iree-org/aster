// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-ilp-scheduler{level=0 ilp-time-limit-ms=5000})))" | FileCheck %s

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // Level 0 (correctness): the CP-SAT model must respect the ds_read -> mfma
  // data dependency (each ds_read produces a fragment the mfma consumes). The
  // hard precedence constraint keeps both reads before the mfma regardless of
  // the source-order-biased objective.
  // CHECK-LABEL: kernel @level0_respects_deps
  // CHECK:         ds_read_b64
  // CHECK:         ds_read_b64
  // CHECK:         v_mfma_f32_16x16x16_f16
  // CHECK:         end_kernel
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
