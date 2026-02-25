// RUN: aster-opt %s --amdgcn-late-waits --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @vmem_load_store
// CHECK:         %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         %[[A:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         %[[B:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         %[[ADDR:.*]] = amdgcn.make_register_range %[[A]], %[[B]]
// CHECK:         amdgcn.load global_load_dword dest %[[DEST]] addr %[[ADDR]]
// CHECK:         amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
// CHECK-NOT:     amdgcn.wait
// CHECK:         amdgcn.store global_store_dword data %[[DEST]] addr %[[ADDR]]
// CHECK:         return
func.func @vmem_load_store() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %3 = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %3 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  %4 = amdgcn.store global_store_dword data %0 addr %3 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  return
}

// -----

// CHECK-LABEL: func.func @shared_load_store
// CHECK:         %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         %[[ADDR:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         amdgcn.load ds_read_b32 dest %[[DEST]] addr %[[ADDR]]
// CHECK:         amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 0
// CHECK-NOT:     amdgcn.wait
// CHECK:         amdgcn.store ds_write_b32 data %[[DEST]] addr %[[ADDR]]
// CHECK:         return
func.func @shared_load_store() {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %token = amdgcn.load ds_read_b32 dest %0 addr %1 : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<?>) -> !amdgcn.read_token<shared>
  %2 = amdgcn.store ds_write_b32 data %0 addr %1 : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<?>) -> !amdgcn.write_token<shared>
  return
}

// -----

// CHECK-LABEL: func.func @vmem_load_branch
// CHECK:         %[[DEST:.*]] = amdgcn.alloca : !amdgcn.vgpr<?>
// CHECK:         amdgcn.load global_load_dword dest %[[DEST]]
// CHECK:         cf.cond_br
// CHECK:       ^bb1:
// CHECK:         amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
// CHECK:         amdgcn.store global_store_dword data %[[DEST]]
// CHECK:       ^bb2:
// CHECK:         amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
// CHECK:         amdgcn.store global_store_dword data %[[DEST]]
// CHECK-NOT:     amdgcn.wait
func.func @vmem_load_branch(%cond: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<?>
  %1 = amdgcn.alloca : !amdgcn.vgpr<?>
  %2 = amdgcn.alloca : !amdgcn.vgpr<?>
  %addr = amdgcn.make_register_range %1, %2 : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  %token = amdgcn.load global_load_dword dest %0 addr %addr : dps(!amdgcn.vgpr<?>) ins(!amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %w0 = amdgcn.store global_store_dword data %0 addr %addr : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb2:
  %w1 = amdgcn.store global_store_dword data %0 addr %addr : ins(!amdgcn.vgpr<?>, !amdgcn.vgpr<[? : ? + 2]>) -> !amdgcn.write_token<flat>
  cf.br ^bb3
^bb3:
  return
}
