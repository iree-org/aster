// RUN: aster-opt %s --amdgcn-late-waits --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @vmem_load_store
// CHECK:         %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         %[[A:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         %[[B:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         %[[ADDR:.*]] = amdgcn.make_register_range %[[A]], %[[B]]
// CHECK:         amdgcn.global_load_dword outs(%[[DEST]]) ins(%[[ADDR]])
// CHECK:         amdgcn.s_waitcnt vmcnt = 0
// CHECK-NOT:     amdgcn.wait
// CHECK:         amdgcn.global_store_dword ins(%[[DEST]], %[[ADDR]])
// CHECK:         return
func.func @vmem_load_store() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %c0_i32_mig1 = arith.constant 0 : i32
  %token = amdgcn.global_load_dword outs(%0) ins(%3) args(%c0_i32_mig1)
      : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) args(i32) -> !amdgcn.read_token<flat>
  %4 = amdgcn.global_store_dword ins(%0, %3) args(%c0_i32_mig1)
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) args(i32) -> !amdgcn.write_token<flat>
  return
}

// -----

// CHECK-LABEL: func.func @shared_load_store
// CHECK:         %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         %[[ADDR:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         amdgcn.ds_read_b32 outs(%[[DEST]]) ins(%[[ADDR]])
// CHECK:         amdgcn.s_waitcnt lgkmcnt = 0
// CHECK-NOT:     amdgcn.wait
// CHECK:         amdgcn.ds_write_b32 ins(%[[ADDR]], %[[DEST]])
// CHECK:         return
func.func @shared_load_store() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %c0_i32_mig1 = arith.constant 0 : i32
  %token = amdgcn.ds_read_b32 outs(%0) ins(%1) args(%c0_i32_mig1)
      : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<?>) args(i32) -> !amdgcn.read_token<shared>
  %2 = amdgcn.ds_write_b32 ins(%1, %0) args(%c0_i32_mig1)
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) args(i32) -> !amdgcn.write_token<shared>
  return
}

// -----

// CHECK-LABEL: func.func @vmem_load_branch
// CHECK:         %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         amdgcn.global_load_dword outs(%[[DEST]])
// CHECK:         cf.cond_br
// CHECK:       ^bb1:
// CHECK:         amdgcn.s_waitcnt vmcnt = 0
// CHECK:         amdgcn.global_store_dword ins(%[[DEST]]
// CHECK:       ^bb2:
// CHECK:         amdgcn.s_waitcnt vmcnt = 0
// CHECK:         amdgcn.global_store_dword ins(%[[DEST]]
// CHECK-NOT:     amdgcn.wait
func.func @vmem_load_branch(%cond: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %c0_i32_mig2 = arith.constant 0 : i32
  %token = amdgcn.global_load_dword outs(%0) ins(%addr) args(%c0_i32_mig2)
      : outs(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) args(i32) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %w0 = amdgcn.global_store_dword ins(%0, %addr) args(%c0_i32_mig2)
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) args(i32) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb2:
  %c0_i32_mig3 = arith.constant 0 : i32
  %w1 = amdgcn.global_store_dword ins(%0, %addr) args(%c0_i32_mig3)
      : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) args(i32) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb3:
  return
}
