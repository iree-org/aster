// RUN: aster-opt --canonicalize --amdgcn-convert-waits %s | FileCheck %s

// CHECK-LABEL:   func.func @test_duplicated_waits() {
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
func.func @test_duplicated_waits() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.vgpr
  %4 = amdgcn.make_register_range %3 : !amdgcn.vgpr
  %result, %token = amdgcn.load global_load_dword dest %4 addr %2 : dps(!amdgcn.vgpr_range<[? + 1]>) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  %5 = amdgcn.store global_store_dword data %result addr %2 : ins(!amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.write_token<flat>
  amdgcn.wait deps %5, %token : !amdgcn.write_token<flat>, !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_pipelined_pattern(
// CHECK:             amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 2
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
func.func @test_pipelined_pattern(%arg0: !amdgcn.vgpr_range<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %result, %token = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %result_0, %token_1 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %result_2, %token_3 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %2:3 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %token, %arg3 = %token_1, %arg4 = %token_3) -> (!amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>) {
    amdgcn.wait deps %arg2 : !amdgcn.read_token<flat>
    %result_4, %token_5 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    scf.yield %arg3, %arg4, %token_5 : !amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2#2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_escaped_waits_1(
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 lgkmcnt = 0
func.func @test_escaped_waits_1(%arg0: !amdgcn.vgpr_range<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  scf.if %arg1 {
    %result, %token = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  }
  amdgcn.wait vm_cnt 0 lgkm_cnt 0
  return
}

// CHECK-LABEL:   func.func @test_if_flow_1(
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
func.func @test_if_flow_1(%arg0: !amdgcn.vgpr_range<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %result, %token = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  } else {
    %result, %token = amdgcn.load global_load_dword dest %1 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_if_flow_2(
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
func.func @test_if_flow_2(%arg0: !amdgcn.vgpr_range<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %result, %token = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    %result_0, %token_1 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    %result_2, %token_3 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    scf.yield %token_3 : !amdgcn.read_token<flat>
  } else {
    %result, %token = amdgcn.load global_load_dword dest %1 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_if_flow_3(
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 1
func.func @test_if_flow_3(%arg0: !amdgcn.vgpr_range<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %result, %token = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    %result_0, %token_1 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    %result_2, %token_3 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  } else {
    %result, %token = amdgcn.load global_load_dword dest %1 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    %result_0, %token_1 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_passthrough_pattern(
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0
func.func @test_passthrough_pattern(%arg0: !amdgcn.vgpr_range<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %result, %token = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %result_0, %token_1 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %result_2, %token_3 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  amdgcn.wait deps %token_3 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_mixed_smem_dsmem(
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 0
func.func @test_mixed_smem_dsmem(%arg0: !amdgcn.sgpr_range<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr_range<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.sgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %result, %token = amdgcn.load s_load_dword dest %0 addr %arg0 : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[? + 2]>) -> !amdgcn.read_token<constant>
  %result_0, %token_1 = amdgcn.load ds_read_b32 dest %1 addr %arg1 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  %result_2, %token_3 = amdgcn.load global_load_dword dest %1 addr %arg2 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<constant>
  return
}

// CHECK-LABEL:   func.func @test_mixed_smem_dsmem_vmem(
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 lgkmcnt = 0
func.func @test_mixed_smem_dsmem_vmem(%arg0: !amdgcn.sgpr_range<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr_range<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.sgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %result, %token = amdgcn.load s_load_dword dest %0 addr %arg0 : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[? + 2]>) -> !amdgcn.read_token<constant>
  %result_0, %token_1 = amdgcn.load ds_read_b32 dest %1 addr %arg1 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
  %result_2, %token_3 = amdgcn.load global_load_dword dest %1 addr %arg2 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<constant>
  amdgcn.wait deps %token_3 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_counts_strength(
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 1
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 2
func.func @test_counts_strength(%arg0: !amdgcn.vgpr_range<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %result, %token = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %result_0, %token_1 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %result_2, %token_3 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait vm_cnt 1 deps %token : !amdgcn.read_token<flat>
  %result_4, %token_5 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %result_6, %token_7 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  %result_8, %token_9 = amdgcn.load global_load_dword dest %0 addr %arg0 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.read_token<flat>
  amdgcn.wait vm_cnt 4 deps %token_5 : !amdgcn.read_token<flat>
  return
}
