// RUN: aster-opt %s --test-hazard-analysis --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: Symbol: test_store_hazard
// CHECK: Op: func.func @test_store_hazard(%{{.*}}: !amdgcn.vgpr<0>, %{{.*}}: !amdgcn.vgpr<[4 : 6]>, %{{.*}}: !amdgcn.vgpr<1>) {...}
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:2, s:0, ds:0}
// CHECK:   }
// CHECK: Op: func.return
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
func.func @test_store_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>) {
  %c0 = arith.constant 0 : i32
  %0 = amdgcn.global_store_dword data %arg0 addr %arg1 offset c(%c0) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// CHECK-LABEL: Symbol: test_store_no_hazard
// CHECK: Op: func.func @test_store_no_hazard(%{{.*}}: !amdgcn.vgpr<0>, %{{.*}}: !amdgcn.vgpr<[4 : 6]>, %{{.*}}: !amdgcn.vgpr<1>) {...}
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_nop
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_nop
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: func.return
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
func.func @test_store_no_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>) {
  // Check that there are no hazards because there are two valu ops in between.
  %c0 = arith.constant 0 : i32
  %0 = amdgcn.global_store_dword data %arg0 addr %arg1 offset c(%c0) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
  amdgcn.v_nop
  amdgcn.v_nop
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// CHECK-LABEL: Symbol: test_cf_hazard
// CHECK: Op: func.func @test_cf_hazard(%{{.*}}: !amdgcn.vgpr<0>, %{{.*}}: !amdgcn.vgpr<[4 : 6]>, %{{.*}}: !amdgcn.vgpr<1>, %{{.*}}: i1) {...}
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: cf.br ^bb2
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:2, s:0, ds:0}
// CHECK:   }
// CHECK: Op: func.return
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// This test checks that hazards propagate through control-flow.
func.func @test_cf_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>, %arg3: i1) {
  cf.cond_br %arg3, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %c0 = arith.constant 0 : i32
  %0 = amdgcn.global_store_dword data %arg0 addr %arg1 offset c(%c0) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// CHECK-LABEL: Symbol: test_cf_diamond_hazard
// CHECK: Op: func.func @test_cf_diamond_hazard(%{{.*}}: !amdgcn.vgpr<0>, %{{.*}}: !amdgcn.vgpr<[4 : 6]>, %{{.*}}: !amdgcn.vgpr<1>, %{{.*}}: i1) {...}
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:2, s:0, ds:0}
// CHECK:   }
// CHECK: Op: cf.br ^bb3
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_nop
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: cf.br ^bb3
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:1, s:0, ds:0}
// CHECK:   }
// CHECK: Op: func.return
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// This test checks that hazards propagate through control-flow, but that counts are optimal.
func.func @test_cf_diamond_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>, %arg3: i1) {
  %c0 = arith.constant 0 : i32
  %0 = amdgcn.global_store_dword data %arg0 addr %arg1 offset c(%c0) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
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
// CHECK-LABEL: Symbol: test_cf_no_hazard
// CHECK: Op: func.func @test_cf_no_hazard(%{{.*}}: !amdgcn.vgpr<0>, %{{.*}}: !amdgcn.vgpr<[4 : 6]>, %{{.*}}: !amdgcn.vgpr<1>, %{{.*}}: i1) {...}
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: cf.cond_br %{{.*}}, ^bb1, ^bb2
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_nop
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: cf.br ^bb2
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_nop
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: func.return
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
func.func @test_cf_no_hazard(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>, %arg3: i1) {
  cf.cond_br %arg3, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %c0 = arith.constant 0 : i32
  %0 = amdgcn.global_store_dword data %arg0 addr %arg1 offset c(%c0) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
  amdgcn.v_nop
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  amdgcn.v_nop
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg2) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// CHECK-LABEL: Symbol: test_hazard_optimality
// CHECK: Op: func.func @test_hazard_optimality(%{{.*}}: !amdgcn.vgpr<0>, %{{.*}}: !amdgcn.vgpr<1>, %{{.*}}: !amdgcn.vgpr<[4 : 6]>, %{{.*}}: !amdgcn.vgpr<1>) {...}
// CHECK:   HAZARD STATE AFTER: <Empty>
// CHECK: Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_write_data_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_store_hazard, %{{.*}} = amdgcn.global_store_dword data %{{.*}} addr %{{.*}} offset c(%{{.*}}) : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>, 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:1, s:0, ds:0}
// CHECK:   }
// CHECK: Op: amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// CHECK: Op: func.return
// CHECK:   HAZARD STATE AFTER: {
// CHECK:     active = [
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_valu_vgpr_readlane_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>), 0, {v:1, s:0, ds:0}},
// CHECK:       {#amdgcn.cdna3_nondlops_valu_mfma_hazard, amdgcn.v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>), 0, {v:2, s:0, ds:0}}
// CHECK:     ]
// CHECK:     nop counts = {v:0, s:0, ds:0}
// CHECK:   }
// This test checks that the second v_mov has no hazards, as the nops required to resolve the first hazard are factored into state for the second v_mov.
func.func @test_hazard_optimality(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<1>, %arg2: !amdgcn.vgpr<[4 : 6]>, %arg3: !amdgcn.vgpr<1>) {
  %c0 = arith.constant 0 : i32
  %0 = amdgcn.global_store_dword data %arg0 addr %arg2 offset c(%c0) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
  %1 = amdgcn.global_store_dword data %arg1 addr %arg2 offset c(%c0) : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
  amdgcn.v_mov_b32 outs(%arg0) ins(%arg3) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
  amdgcn.v_mov_b32 outs(%arg1) ins(%arg3) : outs(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<1>)
  return
}

// -----
// CHECK-LABEL: Symbol: test_store_hazard_const
// This a regression test for a bug where the analysis failed because it didn't handle constants.
func.func @test_store_hazard_const(%arg0: !amdgcn.vgpr<0>, %arg1: !amdgcn.vgpr<[4 : 6]>, %arg2: !amdgcn.vgpr<1>) {
  %c42 = arith.constant 42 : i32
  %c0 = arith.constant 0 : i32
  %0 = amdgcn.global_store_dword data %arg0 addr %arg1 offset c(%c0) : ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<[4 : 6]>) mods(i32) -> !amdgcn.write_token<flat>
  amdgcn.v_mov_b32 outs(%arg0) ins(%c42) : outs(!amdgcn.vgpr<0>) ins(i32)
  return
}

// -----
// expected-error@above {{failed to run hazard analysis}}
func.func @test_non_alloc_registers(%arg0: !amdgcn.vgpr<?>) {
  %c42 = arith.constant 42 : i32
  // expected-error@+2 {{output operands must have allocated semantics}}
  // expected-error@+1 {{failed to get hazards for instruction}}
  amdgcn.v_mov_b32 outs(%arg0) ins(%c42) : outs(!amdgcn.vgpr<?>) ins(i32)
  return
}
