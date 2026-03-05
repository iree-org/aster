// RUN: aster-opt %s --amdgcn-hazards="v_nops=3 s_nops=17" --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_store_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.sopp.sopp <s_nop>, imm = 15
// CHECK:           amdgcn.sopp.sopp <s_nop>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG0]], %[[ARG2]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_store_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>) {
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.vop1.vop1 <v_mov_b32_e32> %arg0, %arg2 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
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
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.sopp.sopp <s_nop>, imm = 15
// CHECK:           amdgcn.sopp.sopp <s_nop>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG0]], %[[ARG2]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_cf_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>, %arg3: i1) {
  cf.cond_br %arg3, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  amdgcn.vop1.vop1 <v_mov_b32_e32> %arg0, %arg2 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
  return
}


// CHECK-LABEL:   func.func @test_no_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<2>) {
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG0]], %[[ARG2]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<2>) -> ()
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.vop1.v_nop
// CHECK:           amdgcn.sopp.sopp <s_nop>, imm = 15
// CHECK:           amdgcn.sopp.sopp <s_nop>
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG1]], %[[ARG2]] : (!amdgcn.vgpr<1>, !amdgcn.vgpr<2>) -> ()
// CHECK:           return
// CHECK:         }
func.func @test_no_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.vgpr<2>) {
  amdgcn.vop1.vop1 <v_mov_b32_e32> %arg0, %arg2 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<2>) -> ()
  amdgcn.vop1.vop1 <v_mov_b32_e32> %arg1, %arg2 : (!amdgcn.vgpr<1>, !amdgcn.vgpr<2>) -> ()
  return
}
