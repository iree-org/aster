// RUN: aster-opt --canonicalize --amdgcn-convert-waits %s | FileCheck %s
// RUN: aster-opt --canonicalize --amdgcn-convert-waits=remove-cf-args=false %s | FileCheck %s --check-prefix=CHECK-KEEP-ARGS

// CHECK-LABEL:   func.func @test_duplicated_waits() {
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
func.func @test_duplicated_waits() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.vgpr
  %4 = amdgcn.make_register_range %3 : !amdgcn.vgpr
  %c0_i32_mig1 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %4 addr %2 offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  %5 = amdgcn.global_store_dword data %result addr %2 offset c(%c0_i32_mig1) : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  amdgcn.wait deps %5, %token : !amdgcn.write_token<flat>, !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_pipelined_pattern(
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             amdgcn.s_waitcnt vmcnt = 2
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
// CHECK-KEEP-ARGS-LABEL:   func.func @test_pipelined_pattern(
// CHECK-KEEP-ARGS: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args{{.*}} -> (!amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>) {
func.func @test_pipelined_pattern(%arg0: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig2 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig2) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig3 = arith.constant 0 : i32
  %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig3) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig4 = arith.constant 0 : i32
  %result_2, %token_3 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig4) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %2:3 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %token, %arg3 = %token_1, %arg4 = %token_3) -> (!amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>) {
    amdgcn.wait deps %arg2 : !amdgcn.read_token<flat>
    %c0_i32_mig5 = arith.constant 0 : i32
    %result_4, %token_5 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig5) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %arg3, %arg4, %token_5 : !amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2#2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_escaped_waits_1(
// CHECK:           amdgcn.s_waitcnt vmcnt = 63 lgkmcnt = 15
func.func @test_escaped_waits_1(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  scf.if %arg1 {
    %c0_i32_mig6 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig6) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  }
  amdgcn.wait vm_cnt 63 lgkm_cnt 15
  return
}

// CHECK-LABEL:   func.func @test_if_flow_1(
// CHECK-KEEP-ARGS-LABEL:   func.func @test_if_flow_1(
// CHECK: scf.if %{{.*}} {
// CHECK-KEEP-ARGS: scf.if %{{.*}} -> (!amdgcn.read_token<flat>) {
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
func.func @test_if_flow_1(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %c0_i32_mig7 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig7) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  } else {
    %c0_i32_mig8 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %1 addr %arg0 offset c(%c0_i32_mig8) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_if_flow_2(
// CHECK-KEEP-ARGS-LABEL:   func.func @test_if_flow_2(
// CHECK: scf.if %{{.*}} {
// CHECK-KEEP-ARGS: scf.if %{{.*}} -> (!amdgcn.read_token<flat>) {
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
func.func @test_if_flow_2(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %c0_i32_mig9 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig9) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig10 = arith.constant 0 : i32
    %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig10) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig11 = arith.constant 0 : i32
    %result_2, %token_3 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig11) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token_3 : !amdgcn.read_token<flat>
  } else {
    %c0_i32_mig12 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %1 addr %arg0 offset c(%c0_i32_mig12) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_if_flow_3(
// CHECK-KEEP-ARGS-LABEL:   func.func @test_if_flow_3(
// CHECK: scf.if %{{.*}} {
// CHECK-KEEP-ARGS: scf.if %{{.*}} -> (!amdgcn.read_token<flat>) {
// CHECK:           amdgcn.s_waitcnt vmcnt = 1
func.func @test_if_flow_3(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %c0_i32_mig13 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig13) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig14 = arith.constant 0 : i32
    %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig14) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig15 = arith.constant 0 : i32
    %result_2, %token_3 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig15) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  } else {
    %c0_i32_mig16 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %1 addr %arg0 offset c(%c0_i32_mig16) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig17 = arith.constant 0 : i32
    %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig17) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_passthrough_pattern(
// CHECK-NOT:       amdgcn.s_waitcnt vmcnt = 2
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
func.func @test_passthrough_pattern(%arg0: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig18 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig18) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig19 = arith.constant 0 : i32
  %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig19) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig20 = arith.constant 0 : i32
  %result_2, %token_3 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig20) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  amdgcn.wait deps %token_3 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_mixed_smem_dsmem(
// CHECK:           amdgcn.s_waitcnt lgkmcnt = 0
func.func @test_mixed_smem_dsmem(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.sgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig1 = arith.constant 0 : i32
  %result, %token = amdgcn.s_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig1) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<constant>
  %result_0, %token_1 = amdgcn.ds_read_b32 dest %1 addr %arg1 offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %c0_i32_mig21 = arith.constant 0 : i32
  %result_2, %token_3 = amdgcn.global_load_dword dest %1 addr %arg2 offset c(%c0_i32_mig21) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<constant>
  return
}

// CHECK-LABEL:   func.func @test_mixed_smem_dsmem_vmem(
// CHECK:           amdgcn.s_waitcnt vmcnt = 0 lgkmcnt = 0
func.func @test_mixed_smem_dsmem_vmem(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.sgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig2 = arith.constant 0 : i32
  %result, %token = amdgcn.s_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig2) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<constant>
  %result_0, %token_1 = amdgcn.ds_read_b32 dest %1 addr %arg1 offset c(%c0_i32_mig2) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %c0_i32_mig22 = arith.constant 0 : i32
  %result_2, %token_3 = amdgcn.global_load_dword dest %1 addr %arg2 offset c(%c0_i32_mig22) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<constant>
  amdgcn.wait deps %token_3 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_counts_strength(
// CHECK:           amdgcn.s_waitcnt vmcnt = 1
// CHECK:           amdgcn.s_waitcnt vmcnt = 2
func.func @test_counts_strength(%arg0: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig23 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig23) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig24 = arith.constant 0 : i32
  %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig24) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig25 = arith.constant 0 : i32
  %result_2, %token_3 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig25) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait vm_cnt 1 deps %token : !amdgcn.read_token<flat>
  %c0_i32_mig26 = arith.constant 0 : i32
  %result_4, %token_5 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig26) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig27 = arith.constant 0 : i32
  %result_6, %token_7 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig27) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig28 = arith.constant 0 : i32
  %result_8, %token_9 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig28) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait vm_cnt 4 deps %token_5 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_mixed_iter_args(
// CHECK-SAME:      %[[ADDR:.*]]: !amdgcn.vgpr<[? + 2]>) -> index {
// CHECK:           %[[RESULT:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (index) {
// CHECK-NOT:         !amdgcn.read_token
// CHECK:             amdgcn.s_waitcnt vmcnt = 0
// CHECK:             scf.yield %{{.*}} : index
// CHECK:           return %[[RESULT]] : index
// CHECK-KEEP-ARGS-LABEL:   func.func @test_mixed_iter_args(
// CHECK-KEEP-ARGS:   scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!amdgcn.read_token<flat>, index) {
func.func @test_mixed_iter_args(%arg0: !amdgcn.vgpr<[? + 2]>) -> index {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig29 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig29) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %out:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%tok = %token, %acc = %c0) -> (!amdgcn.read_token<flat>, index) {
    amdgcn.wait deps %tok : !amdgcn.read_token<flat>
    %c0_i32_mig30 = arith.constant 0 : i32
    %result2, %token2 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig30) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %new_acc = arith.addi %acc, %c1 : index
    scf.yield %token2, %new_acc : !amdgcn.read_token<flat>, index
  }
  amdgcn.wait deps %out#0 : !amdgcn.read_token<flat>
  return %out#1 : index
}

// CHECK-LABEL:   func.func @cf_args(
// CHECK:           %[[POISON:.*]] = ub.poison : !amdgcn.read_token<flat>
// CHECK: ^{{.*}}:
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
// CHECK: ^{{.*}}:
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
// CHECK:           return %[[POISON]] : !amdgcn.read_token<flat>
// CHECK-KEEP-ARGS-LABEL:   func.func @cf_args(
// CHECK-KEEP-ARGS: cf.cond_br{{.*}}
// CHECK-KEEP-ARGS: ^{{.*}}(%{{.*}}: !amdgcn.read_token<flat>):
// CHECK-KEEP-ARGS: cf.cond_br{{.*}}
// CHECK-KEEP-ARGS: ^{{.*}}(%{{.*}}: !amdgcn.read_token<flat>):
func.func @cf_args(%arg0: !amdgcn.vgpr<[? + 2]>, %cond: i1)  -> !amdgcn.read_token<flat> {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig31 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig31) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1(%token : !amdgcn.read_token<flat>), ^bb2(%token : !amdgcn.read_token<flat>)
^bb1(%1: !amdgcn.read_token<flat>):
  amdgcn.wait deps %1 : !amdgcn.read_token<flat>
  %c0_i32_mig32 = arith.constant 0 : i32
  %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig32) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  cf.cond_br %cond, ^bb1(%token_1 : !amdgcn.read_token<flat>), ^bb2 (%token_1 : !amdgcn.read_token<flat>)
^bb2(%2: !amdgcn.read_token<flat>):
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  %3 = scf.if %cond -> (!amdgcn.read_token<flat>) {
    %c0_i32_mig33 = arith.constant 0 : i32
    %result_2, %token_2 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig33) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token_2 : !amdgcn.read_token<flat>
  } else {
    %c0_i32_mig34 = arith.constant 0 : i32
    %result_2, %token_2 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig34) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token_2 : !amdgcn.read_token<flat>
  }
  return %3 : !amdgcn.read_token<flat>
}

// -----
// Tests for ops that consume poison token values.
// -----

// CHECK-LABEL:   func.func @wait_on_poison_read_token() {
// CHECK:           %[[POISON:.*]] = ub.poison : !amdgcn.read_token<flat>
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
// CHECK:           return
func.func @wait_on_poison_read_token() {
  %poison = ub.poison : !amdgcn.read_token<flat>
  amdgcn.wait deps %poison : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @wait_on_poison_write_token() {
// CHECK:           %[[POISON:.*]] = ub.poison : !amdgcn.write_token<flat>
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
// CHECK:           return
func.func @wait_on_poison_write_token() {
  %poison = ub.poison : !amdgcn.write_token<flat>
  amdgcn.wait deps %poison : !amdgcn.write_token<flat>
  return
}

// CHECK-LABEL:   func.func @wait_on_poison_lgkm() {
// CHECK:           amdgcn.s_waitcnt lgkmcnt = 0
// CHECK:           return
func.func @wait_on_poison_lgkm() {
  %poison_const = ub.poison : !amdgcn.read_token<constant>
  %poison_shared = ub.poison : !amdgcn.read_token<shared>
  amdgcn.wait deps %poison_const, %poison_shared : !amdgcn.read_token<constant>, !amdgcn.read_token<shared>
  return
}

// CHECK-LABEL:   func.func @wait_on_mixed_real_and_poison(
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
// CHECK:           return
func.func @wait_on_mixed_real_and_poison(%arg0: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig35 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig35) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %poison = ub.poison : !amdgcn.read_token<flat>
  amdgcn.wait deps %token, %poison : !amdgcn.read_token<flat>, !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @for_with_poison_init(
// CHECK:           scf.for {{.*}} {
// CHECK:             amdgcn.s_waitcnt vmcnt = 0
// CHECK:           }
// CHECK:           amdgcn.s_waitcnt vmcnt = 0
// CHECK:           return
func.func @for_with_poison_init(%arg0: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %poison = ub.poison : !amdgcn.read_token<flat>
  %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%tok = %poison) -> (!amdgcn.read_token<flat>) {
    amdgcn.wait deps %tok : !amdgcn.read_token<flat>
    %c0_i32_mig36 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig36) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %out : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL:   func.func @test_wait_canonicalization(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.ds_read_b32 dest %[[ALLOCA_0]] addr %[[ARG0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.ds_read_b32 dest %[[ALLOCA_0]] addr %[[ARG0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
// CHECK:           %[[VAL_2:.*]], %[[LOAD_2:.*]] = amdgcn.ds_read_b32 dest %[[ALLOCA_0]] addr %[[ARG0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
// CHECK:           %[[VAL_3:.*]], %[[LOAD_3:.*]] = amdgcn.ds_read_b32 dest %[[ALLOCA_0]] addr %[[ARG0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
// CHECK:           amdgcn.s_waitcnt lgkmcnt = 2
// CHECK:           amdgcn.test_inst : () -> ()
// CHECK:           amdgcn.s_waitcnt lgkmcnt = 1
// CHECK:           amdgcn.test_inst : () -> ()
// CHECK:           amdgcn.s_waitcnt lgkmcnt = 0
// CHECK:           return
// CHECK:         }
func.func @test_wait_canonicalization(%addr: !amdgcn.vgpr) {
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig3 = arith.constant 0 : i32
  %result_0, %token = amdgcn.ds_read_b32 dest %1 addr %addr offset c(%c0_i32_mig3) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig4 = arith.constant 0 : i32
  %result_1, %token_1 = amdgcn.ds_read_b32 dest %1 addr %addr offset c(%c0_i32_mig4) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %3 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig5 = arith.constant 0 : i32
  %result_2, %token_2 = amdgcn.ds_read_b32 dest %1 addr %addr offset c(%c0_i32_mig5) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %4 = amdgcn.alloca : !amdgcn.vgpr
  %c0_i32_mig6 = arith.constant 0 : i32
  %result_3, %token_3 = amdgcn.ds_read_b32 dest %1 addr %addr offset c(%c0_i32_mig6) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  // These 2 waits can be canonicalized into a single wait with lgkmcnt = 2.
  amdgcn.wait deps %token : !amdgcn.read_token<shared>
  amdgcn.wait deps %token_1 : !amdgcn.read_token<shared>
  amdgcn.test_inst : () -> ()
  amdgcn.wait deps %token_2 : !amdgcn.read_token<shared>
  amdgcn.test_inst : () -> ()
  amdgcn.wait deps %token_3 : !amdgcn.read_token<shared>
  return
}
