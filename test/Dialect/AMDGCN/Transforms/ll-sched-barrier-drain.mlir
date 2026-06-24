// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-low-level-scheduler{debug-stalls=false})))" | FileCheck %s

!v = !amdgcn.vgpr

amdgcn.module @test target = #amdgcn.target<gfx942> {

  // barrier has no cross-thread visible predecessors; reads and mfma may
  // reorder freely around it.
  // CHECK-LABEL: kernel @barrier_first_then_mfma
  // CHECK:         barrier
  // CHECK:         ds_read_b64
  // CHECK:         ds_read_b64
  // CHECK:         v_mfma_f32_16x16x16_f16
  // CHECK:         end_kernel
  amdgcn.kernel @barrier_first_then_mfma {
    %addr0 = amdgcn.alloca : !v
    %addr1 = amdgcn.alloca : !v
    %rd0 = amdgcn.alloca : !v
    %rd1 = amdgcn.alloca : !v
    %d0 = amdgcn.make_register_range %rd0, %rd1 : !v, !v
    %rd2 = amdgcn.alloca : !v
    %rd3 = amdgcn.alloca : !v
    %d1 = amdgcn.make_register_range %rd2, %rd3 : !v, !v
    %acc0 = amdgcn.alloca : !v
    %acc1 = amdgcn.alloca : !v
    %acc2 = amdgcn.alloca : !v
    %acc3 = amdgcn.alloca : !v
    %acc = amdgcn.make_register_range %acc0, %acc1, %acc2, %acc3 : !v, !v, !v, !v
    %c0 = arith.constant 0 : i32
    %r0, %t0 = amdgcn.ds_read_b64 dest %d0 addr %addr0 offset c(%c0) : outs(!amdgcn.vgpr<[? + 2]>) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %r1, %t1 = amdgcn.ds_read_b64 dest %d1 addr %addr1 offset c(%c0) : outs(!amdgcn.vgpr<[? + 2]>) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %res = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc) ins(%r0, %r1, %acc) : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
    amdgcn.barrier scope(<workgroup>)
    amdgcn.end_kernel
  }

  // barrier does not pin unrelated ds_reads; mfma still follows its operands.
  // CHECK-LABEL: kernel @two_barriers
  // CHECK:         barrier
  // CHECK:         barrier
  // CHECK:         ds_read_b64
  // CHECK:         v_mfma_f32_16x16x16_f16
  // CHECK:         ds_read_b64
  // CHECK:         v_mfma_f32_16x16x16_f16
  // CHECK:         end_kernel
  amdgcn.kernel @two_barriers {
    %addr0 = amdgcn.alloca : !v
    %addr1 = amdgcn.alloca : !v
    %rd0 = amdgcn.alloca : !v
    %rd1 = amdgcn.alloca : !v
    %d0 = amdgcn.make_register_range %rd0, %rd1 : !v, !v
    %rd2 = amdgcn.alloca : !v
    %rd3 = amdgcn.alloca : !v
    %d1 = amdgcn.make_register_range %rd2, %rd3 : !v, !v
    %a_acc0 = amdgcn.alloca : !v
    %a_acc1 = amdgcn.alloca : !v
    %a_acc2 = amdgcn.alloca : !v
    %a_acc3 = amdgcn.alloca : !v
    %acc_a = amdgcn.make_register_range %a_acc0, %a_acc1, %a_acc2, %a_acc3 : !v, !v, !v, !v
    %b_acc0 = amdgcn.alloca : !v
    %b_acc1 = amdgcn.alloca : !v
    %b_acc2 = amdgcn.alloca : !v
    %b_acc3 = amdgcn.alloca : !v
    %acc_b = amdgcn.make_register_range %b_acc0, %b_acc1, %b_acc2, %b_acc3 : !v, !v, !v, !v
    %c0 = arith.constant 0 : i32
    %r0, %t0 = amdgcn.ds_read_b64 dest %d0 addr %addr0 offset c(%c0) : outs(!amdgcn.vgpr<[? + 2]>) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %resa = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc_a) ins(%r0, %r0, %acc_a) : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
    amdgcn.barrier scope(<workgroup>)
    %r1, %t1 = amdgcn.ds_read_b64 dest %d1 addr %addr1 offset c(%c0) : outs(!amdgcn.vgpr<[? + 2]>) ins(!v) mods(i32) -> !amdgcn.read_token<shared>
    %resb = amdgcn.v_mfma_f32_16x16x16_f16 outs(%acc_b) ins(%r1, %r1, %acc_b) : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>)
    amdgcn.barrier scope(<workgroup>)
    amdgcn.end_kernel
  }

  // CHECK-LABEL: kernel @token_barrier_order
  // CHECK:         ds_write_b32
  // CHECK:         token_barrier
  // CHECK:         ds_read_b32
  // CHECK:         end_kernel
  amdgcn.kernel @token_barrier_order {
    %addr = amdgcn.alloca : !v
    %data = amdgcn.alloca : !v
    %rd = amdgcn.alloca : !v
    %c0 = arith.constant 0 : i32
    %wtok = amdgcn.ds_write_b32 data %data addr %addr offset c(%c0) : ins(!v, !v) mods(i32) -> !amdgcn.write_token<shared>
    // Tokenized barrier with fence token keeps write-before-read ordering.
    %bar = amdgcn.token_barrier scope(<workgroup>) deps %wtok : !amdgcn.write_token<shared>
    %r, %t = amdgcn.ds_read_b32 dest %rd addr %addr offset c(%c0) : outs(!v) ins(!v) mods(i32) -> !amdgcn.read_token<shared> fence_token %bar : !amdgcn.fence_token
    amdgcn.end_kernel
  }
}
