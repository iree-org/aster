// RUN: aster-opt --cse %s | FileCheck %s

// CHECK-LABEL:   func.func @unallocated_registers() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           return %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }
func.func @unallocated_registers() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  return %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @allocated_registers() -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           return %[[ALLOCA_0]], %[[ALLOCA_0]] : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
// CHECK:         }
func.func @allocated_registers() -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<1>
  return %0, %1 : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
}

// CHECK-LABEL:   func.func @csed_mov() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.v_mov_b32 outs(%[[ALLOCA_1]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
// CHECK:           return %[[VAL_0]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }
func.func @csed_mov() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
  %3 = amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
  return %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @uncsed_mov() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.v_mov_b32 outs(%[[ALLOCA_1]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
// CHECK:           %[[VAL_1:.*]] = amdgcn.v_mov_b32 outs(%[[ALLOCA_2]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }
func.func @uncsed_mov() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
  %4 = amdgcn.v_mov_b32 outs(%2) ins(%0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
  return %3, %4 : !amdgcn.vgpr, !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @chained_mov() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.v_mov_b32 outs(%[[ALLOCA_1]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
// CHECK:           %[[VAL_1:.*]] = amdgcn.v_mov_b32 outs(%[[VAL_0]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }
func.func @chained_mov() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
  %3 = amdgcn.v_mov_b32 outs(%2) ins(%0) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr)
  return %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @allocated_csed_mov() -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<2>) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_1]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_1]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
// CHECK:           return %[[ALLOCA_1]], %[[ALLOCA_1]] : !amdgcn.vgpr<2>, !amdgcn.vgpr<2>
// CHECK:         }
func.func @allocated_csed_mov() -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<2>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<2>
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
  return %1, %1 : !amdgcn.vgpr<2>, !amdgcn.vgpr<2>
}

// CHECK-LABEL:   func.func @allocated_uncsed_mov() -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr<3>
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_1]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_2]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<1>)
// CHECK:           return %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
// CHECK:         }
func.func @allocated_uncsed_mov() -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<2>
  %2 = amdgcn.alloca : !amdgcn.vgpr<3>
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_mov_b32 outs(%2) ins(%0) : outs(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<1>)
  return %1, %2 : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
}

// CHECK-LABEL:   func.func @allocated_chained_mov() -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<2>) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr<2>
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_1]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_1]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
// CHECK:           return %[[ALLOCA_1]], %[[ALLOCA_1]] : !amdgcn.vgpr<2>, !amdgcn.vgpr<2>
// CHECK:         }
func.func @allocated_chained_mov() -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<2>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<2>
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<1>)
  return %1, %1 : !amdgcn.vgpr<2>, !amdgcn.vgpr<2>
}

// CHECK-LABEL:   func.func @same_allocated_csed_mov() -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_0]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_0]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
// CHECK:           return %[[ALLOCA_0]], %[[ALLOCA_0]] : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
// CHECK:         }
func.func @same_allocated_csed_mov() -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<1>
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
  return %1, %1 : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
}

// CHECK-LABEL:   func.func @same_allocated_uncsed_mov() -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_0]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_0]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
// CHECK:           return %[[ALLOCA_0]], %[[ALLOCA_0]] : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
// CHECK:         }
func.func @same_allocated_uncsed_mov() -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<1>
  %2 = amdgcn.alloca : !amdgcn.vgpr<1>
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_mov_b32 outs(%2) ins(%0) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
  return %1, %2 : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
}

// CHECK-LABEL:   func.func @same_allocated_chained_mov() -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr<1>
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_0]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_mov_b32 outs(%[[ALLOCA_0]]) ins(%[[ALLOCA_0]]) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
// CHECK:           return %[[ALLOCA_0]], %[[ALLOCA_0]] : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
// CHECK:         }
func.func @same_allocated_chained_mov() -> (!amdgcn.vgpr<1>, !amdgcn.vgpr<1>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<1>
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_mov_b32 outs(%1) ins(%0) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
  return %1, %1 : !amdgcn.vgpr<1>, !amdgcn.vgpr<1>
}


amdgcn.module @mod target = <gfx942> {
// CHECK-LABEL: kernel @gpu_copy_kernel_unchecked
//      CHECK:    %[[R7:.*]] = alloca : !amdgcn.vgpr<10>
// CHECK-NEXT:    %[[R8:.*]] = alloca : !amdgcn.vgpr<11>
// CHECK-NEXT:    v_mov_b32 outs(%[[R7]]) ins(%{{.*}}) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.sgpr<2>)
// CHECK-NEXT:    v_mov_b32 outs(%[[R8]]) ins(%{{.*}}) : outs(!amdgcn.vgpr<11>) ins(!amdgcn.sgpr<3>)
// CHECK-NEXT:    %[[R11:.*]] = make_register_range %[[R7]], %[[R8]] : !amdgcn.vgpr<10>, !amdgcn.vgpr<11>
// CHECK-NEXT:    %[[R12:.*]] = alloca : !amdgcn.vgpr<12>
// CHECK-NEXT:    %[[R13:.*]] = make_register_range %[[R12]] : !amdgcn.vgpr<12>
// CHECK-NEXT:    %{{.*}} = load global_load_dword dest %[[R13]] addr %[[R11]] : dps(!amdgcn.vgpr<12>) ins(!amdgcn.vgpr<[10 : 12]>) -> !amdgcn.read_token<flat>
// CHECK-NEXT:    %{{.*}} = load s_load_dwordx2 dest %{{.*}} addr %{{.*}} : dps(!amdgcn.sgpr<[2 : 4]>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<constant>
// CHECK-NEXT:    amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
//
// We want to make sure R7, R8 and R11 are not reused after CSE: a proper liveness analysis is needed.
// CHECK-NEXT:    v_mov_b32 outs(%[[R7]]) ins(%{{.*}}) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.sgpr<2>)
// CHECK-NEXT:    v_mov_b32 outs(%[[R8]]) ins(%{{.*}}) : outs(!amdgcn.vgpr<11>) ins(!amdgcn.sgpr<3>)
// CHECK-NEXT:    %{{.*}} = store global_store_dword data %[[R13]] addr %[[R11]] : ins(!amdgcn.vgpr<12>, !amdgcn.vgpr<[10 : 12]>) -> !amdgcn.write_token<flat>
// CHECK-NEXT:    end_kernel
  kernel @gpu_copy_kernel_unchecked {
    %0 = alloca : !amdgcn.sgpr<0>
    %1 = alloca : !amdgcn.sgpr<1>
    %2 = make_register_range %0, %1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %3 = alloca : !amdgcn.sgpr<2>
    %4 = alloca : !amdgcn.sgpr<3>
    %5 = make_register_range %3, %4 : !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %t1 = amdgcn.load s_load_dwordx2 dest %5 addr %2 : dps(!amdgcn.sgpr<[2 : 4]>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<constant>
    amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
    %7 = alloca : !amdgcn.vgpr<10>
    %8 = alloca : !amdgcn.vgpr<11>
    amdgcn.v_mov_b32 outs(%7) ins(%3) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.sgpr<2>)
    amdgcn.v_mov_b32 outs(%8) ins(%4) : outs(!amdgcn.vgpr<11>) ins(!amdgcn.sgpr<3>)
    %11 = make_register_range %7, %8 : !amdgcn.vgpr<10>, !amdgcn.vgpr<11>
    %12 = alloca : !amdgcn.vgpr<12>
    %13 = make_register_range %12 : !amdgcn.vgpr<12>
    %t2 = amdgcn.load global_load_dword dest %13 addr %11 : dps(!amdgcn.vgpr<12>) ins(!amdgcn.vgpr<[10 : 12]>) -> !amdgcn.read_token<flat>
    %15 = alloca : !amdgcn.sgpr<0>
    %16 = alloca : !amdgcn.sgpr<1>
    %17 = make_register_range %15, %16 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %18 = alloca : !amdgcn.sgpr<2>
    %19 = alloca : !amdgcn.sgpr<3>
    %20 = make_register_range %18, %19 : !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %t3 = amdgcn.load s_load_dwordx2 dest %20 addr %17 : dps(!amdgcn.sgpr<[2 : 4]>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<constant>
    amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
    %22 = alloca : !amdgcn.vgpr<10>
    %23 = alloca : !amdgcn.vgpr<11>
    amdgcn.v_mov_b32 outs(%22) ins(%18) : outs(!amdgcn.vgpr<10>) ins(!amdgcn.sgpr<2>)
    amdgcn.v_mov_b32 outs(%23) ins(%19) : outs(!amdgcn.vgpr<11>) ins(!amdgcn.sgpr<3>)
    %26 = make_register_range %22, %23 : !amdgcn.vgpr<10>, !amdgcn.vgpr<11>
    %t4 = amdgcn.store global_store_dword data %13 addr %26 : ins(!amdgcn.vgpr<12>, !amdgcn.vgpr<[10 : 12]>) -> !amdgcn.write_token<flat>
    end_kernel
  }
}
