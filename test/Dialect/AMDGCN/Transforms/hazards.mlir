// RUN: aster-opt %s --amdgcn-hazards --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_store_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_store_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>) {
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// CHECK-LABEL:   func.func @test_store_no_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_store_no_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>) {
  // Check that there are no hazards because there are two valu ops in between.
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_nop
  amdgcn.v_nop
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// This test checks that hazards propagate through control-flow.
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

// -----
// This test checks that hazards propagate through control-flow, but that counts are optimal.
// CHECK-LABEL:   func.func @test_cf_diamond_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG3:.*]]: i1) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           cf.cond_br %[[ARG3]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           amdgcn.v_nop
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_cf_diamond_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>, %arg3: i1) {
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  cf.cond_br %arg3, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  cf.br ^bb3
^bb2:  // pred: ^bb0
  amdgcn.v_nop
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// CHECK-LABEL:   func.func @test_cf_no_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG3:.*]]: i1) {
// CHECK:           cf.cond_br %[[ARG3]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.v_nop
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_cf_no_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>, %arg3: i1) {
  cf.cond_br %arg3, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg1 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_nop
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  amdgcn.v_nop
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// This test checks that the second v_mov has no hazards, as the nops required to resolve the first hazard are factored into state for the second v_mov.
// CHECK-LABEL:   func.func @test_hazard_optimality(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG3:.*]]: !amdgcn.vgpr<1>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG2]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           %[[STORE_1:.*]] = amdgcn.store global_store_dword data %[[ARG1]] addr %[[ARG2]] : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG3]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG1]]) ins(%[[ARG3]]) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_hazard_optimality(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.vgpr<[4 : 6]>, %arg3: !amdgcn.vgpr<1>) {
  %0 = amdgcn.store global_store_dword data %arg0 addr %arg2 : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  %1 = amdgcn.store global_store_dword data %arg1 addr %arg2 : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg3) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_mov_b32 outs(%arg1) ins(%arg3) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
  return
}

// -----

// CHECK-LABEL:   func.func @test_backedge_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG3:.*]]: i1) {
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           cf.cond_br %[[ARG3]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_backedge_hazard(%data: !amdgcn.vgpr<0>, %addr: !amdgcn.vgpr<[4 : 6]>, %value: !amdgcn.vgpr<1>, %cond: i1) {
  cf.br ^bb1
^bb1:
  amdgcn.v_mov_b32 outs(%data) ins(%value) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  %token = amdgcn.store global_store_dword data %data addr %addr : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  cf.cond_br %cond , ^bb1, ^bb2
^bb2:
  amdgcn.v_mov_b32 outs(%data) ins(%value) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// CHECK-LABEL:   func.func @test_backedge_no_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG3:.*]]: i1) {
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.v_nop
// CHECK:           cf.cond_br %[[ARG3]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_backedge_no_hazard(%data: !amdgcn.vgpr<0>, %addr: !amdgcn.vgpr<[4 : 6]>, %value: !amdgcn.vgpr<1>, %cond: i1) {
  cf.br ^bb1
^bb1:
  amdgcn.v_nop
  amdgcn.v_mov_b32 outs(%data) ins(%value) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  %token = amdgcn.store global_store_dword data %data addr %addr : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_nop
  cf.cond_br %cond , ^bb1, ^bb2
^bb2:
  amdgcn.v_nop
  amdgcn.v_mov_b32 outs(%data) ins(%value) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// CHECK-LABEL:   func.func @test_backedge_exit_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG3:.*]]: i1) {
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.v_nop
// CHECK:           cf.cond_br %[[ARG3]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_backedge_exit_hazard(%data: !amdgcn.vgpr<0>, %addr: !amdgcn.vgpr<[4 : 6]>, %value: !amdgcn.vgpr<1>, %cond: i1) {
  cf.br ^bb1
^bb1:
  amdgcn.v_nop
  amdgcn.v_mov_b32 outs(%data) ins(%value) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  %token = amdgcn.store global_store_dword data %data addr %addr : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_nop
  cf.cond_br %cond , ^bb1, ^bb2
^bb2:
  amdgcn.v_mov_b32 outs(%data) ins(%value) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// CHECK-LABEL:   func.func @test_backedge_no_exit_hazard(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<0>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr<[4 : 6]>,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.vgpr<1>,
// CHECK-SAME:      %[[ARG3:.*]]: i1) {
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.v_nop
// CHECK:           amdgcn.v_nop
// CHECK:           cf.cond_br %[[ARG3]], ^bb1, ^bb2
// CHECK:         ^bb2:
// CHECK:           amdgcn.v_mov_b32 outs(%[[ARG0]]) ins(%[[ARG2]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:           return
// CHECK:         }
func.func @test_backedge_no_exit_hazard(%data: !amdgcn.vgpr<0>, %addr: !amdgcn.vgpr<[4 : 6]>, %value: !amdgcn.vgpr<1>, %cond: i1) {
  cf.br ^bb1
^bb1:
  amdgcn.v_mov_b32 outs(%data) ins(%value) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  %token = amdgcn.store global_store_dword data %data addr %addr : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) -> !amdgcn.write_token<flat>
  amdgcn.v_nop
  amdgcn.v_nop
  cf.cond_br %cond , ^bb1, ^bb2
^bb2:
  amdgcn.v_mov_b32 outs(%data) ins(%value) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}
