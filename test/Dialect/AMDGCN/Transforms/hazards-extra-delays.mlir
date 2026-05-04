// RUN: aster-opt %s --amdgcn-hazards="v_nops=3 s_nops=17" --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_store_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.s_nop 15
// CHECK:           amdgcn.s_nop 0
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_store_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>) {
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// CHECK-LABEL:   func.func @test_cf_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG3:.*]]: i1) {
// CHECK:           cf.cond_br %[[ARG3]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.s_nop 15
// CHECK:           amdgcn.s_nop 0
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_cf_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>, %arg3: i1) {
  cf.cond_br %arg3, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}


// CHECK-LABEL:   func.func @test_no_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<2>) {
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<2>)
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.s_nop 15
// CHECK:           amdgcn.s_nop 0
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG1]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<2>)
// CHECK:           return
// CHECK:         }
func.func @test_no_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.vgpr<2>) {
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<2>)
  amdgcn.v_mov_b32 outs(%arg1) ins(%arg2) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<2>)
  return
}
