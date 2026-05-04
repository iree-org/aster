// RUN: aster-opt %s --test-wait-analysis | FileCheck %s

// CHECK-LABEL: test_duplicated_waits
// CHECK:       Op: func.func @test_duplicated_waits() {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: %[[R_0:.*]], %[[R_1:.*]] = amdgcn.global_load_dword dest %[[R_2:.*]] addr %[[R_3:.*]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_1]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait deps %[[R_1]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_1]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_1]], {{[0-9]*}}, 0, flat}]}
// CHECK:       Op: amdgcn.wait deps %[[R_1]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_1]], {{[0-9]*}}, 0, flat}]}
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: nowait, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: []}
// CHECK:       Op: %[[R_4:.*]] = amdgcn.global_store_dword data %[[R_0]] addr %[[R_3]] offset c(%{{.*}}) : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {vm_cnt: nowait, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: []}
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait deps %[[R_4]], %[[R_1]] : !amdgcn.write_token<flat>, !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_4]], {{[0-9]*}}, 0, flat}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_4]], {{[0-9]*}}, 0, flat}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_duplicated_waits() {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %2 = amdgcn.make_register_range %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.vgpr
  %4 = amdgcn.make_register_range %3 : !amdgcn.vgpr
  %c0_i32_mig1 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %4 addr %2 offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  // Wait again on the same token, so the second wait is redundant.
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  %5 = amdgcn.global_store_dword data %result addr %2 offset c(%c0_i32_mig1) : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.write_token<flat>
  amdgcn.wait deps %5, %token : !amdgcn.write_token<flat>, !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL: test_pipelined_pattern
// CHECK:       Op: func.func @test_pipelined_pattern(%[[R_0:.*]]: !amdgcn.vgpr<[? + 2]>) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: %[[R_1:.*]], %[[R_2:.*]] = amdgcn.global_load_dword dest %[[R_3:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_4:.*]], %[[R_5:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 1, flat}, {%[[R_5]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_6:.*]], %[[R_7:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 1, flat}, {%[[R_5]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_8:.*]]:3 = scf.for %[[R_9:.*]] = %[[R_10:.*]] to %[[R_11:.*]] step %[[R_12:.*]] iter_args(%[[R_13:.*]] = %[[R_2]], %[[R_14:.*]] = %[[R_5]], %[[R_15:.*]] = %[[R_7]]) -> (!amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>) {...}
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_16:.*]], {{[0-9]*}}, 0, flat}, {%[[R_17:.*]], {{[0-9]*}}, 2, flat}, {%[[R_18:.*]], {{[0-9]*}}, 1, flat}]
// CHECK:       Op: amdgcn.wait deps %[[R_13]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}, {%[[R_13]], {{[0-9]*}}, 2, flat}, {%[[R_14]], {{[0-9]*}}, 1, flat}, {%[[R_15]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}, {%[[R_14]], {{[0-9]*}}, 1, flat}, {%[[R_15]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: 2, lgkm_cnt: nowait}, waited_tokens: [{%[[R_13]], {{[0-9]*}}, 2, flat}], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_13]], {{[0-9]*}}, 2, flat}]}
// CHECK:       Op: %{{.*}} = arith.constant 0 : i32
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}, {%[[R_14]], {{[0-9]*}}, 1, flat}, {%[[R_15]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: 2, lgkm_cnt: nowait}, waited_tokens: [{%[[R_13]], {{[0-9]*}}, 2, flat}], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_13]], {{[0-9]*}}, 2, flat}]}
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}, {%[[R_14]], {{[0-9]*}}, 1, flat}, {%[[R_15]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_19:.*]], %[[R_20:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}, {%[[R_14]], {{[0-9]*}}, 1, flat}, {%[[R_15]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_14]], {{[0-9]*}}, 2, flat}, {%[[R_15]], {{[0-9]*}}, 1, flat}, {%[[R_20]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: scf.yield %[[R_14]], %[[R_15]], %[[R_20]] : !amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_14]], {{[0-9]*}}, 2, flat}, {%[[R_15]], {{[0-9]*}}, 1, flat}, {%[[R_20]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_14]], {{[0-9]*}}, 2, flat}, {%[[R_15]], {{[0-9]*}}, 1, flat}, {%[[R_20]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait deps %[[R_16]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_16]], {{[0-9]*}}, 0, flat}, {%[[R_17]], {{[0-9]*}}, 2, flat}, {%[[R_18]], {{[0-9]*}}, 1, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_5]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_16]], {{[0-9]*}}, 0, flat}, {%[[R_17]], {{[0-9]*}}, 2, flat}, {%[[R_18]], {{[0-9]*}}, 1, flat}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_5]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_16]], {{[0-9]*}}, 0, flat}, {%[[R_17]], {{[0-9]*}}, 2, flat}, {%[[R_18]], {{[0-9]*}}, 1, flat}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
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
  // There's always at least 2 outstanding loads in the pipeline.
  %2:3 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %token, %arg3 = %token_1, %arg4 = %token_3) -> (!amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>) {
    amdgcn.wait deps %arg2 : !amdgcn.read_token<flat>
    %c0_i32_mig5 = arith.constant 0 : i32
    %result_4, %token_5 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig5) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %arg3, %arg4, %token_5 : !amdgcn.read_token<flat>, !amdgcn.read_token<flat>, !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2#2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL: test_escaped_waits_1
// CHECK:       Op: func.func @test_escaped_waits_1(%[[R_0:.*]]: !amdgcn.vgpr<[? + 2]>) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: scf.for %[[R_1:.*]] = %[[R_2:.*]] to %[[R_3:.*]] step %[[R_4:.*]] {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{<escaped>, 0, flat}]
// CHECK:       Op: %[[R_5:.*]], %[[R_6:.*]] = amdgcn.global_load_dword dest %[[R_7:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{<escaped>, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_6]], {{[0-9]*}}, 0, flat}, {<escaped>, 1, flat}]
// CHECK:       Op: scf.yield
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_6]], {{[0-9]*}}, 0, flat}, {<escaped>, 1, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_6]], {{[0-9]*}}, 0, flat}, {<escaped>, 1, flat}]
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{<escaped>, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{<escaped>, 0, flat}]
func.func @test_escaped_waits_1(%arg0: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // The load's token escapes the loop.
  scf.for %arg1 = %c0 to %c4 step %c1 {
    %c0_i32_mig6 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig6) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  }
  return
}

// CHECK-LABEL: test_escaped_waits_2
// CHECK:       Op: func.func @test_escaped_waits_2(%[[R_0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[R_1:.*]]: i1) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: scf.if %[[R_1]] {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{<escaped>, 0, flat}]
// CHECK:       Op: %[[R_2:.*]], %[[R_3:.*]] = amdgcn.global_load_dword dest %[[R_4:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_3]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: scf.yield
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_3]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_3]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait vm_cnt 0 lgkm_cnt 0
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{<escaped>, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: 0}, waited_tokens: [], implied_tokens: [{<escaped>, 0, flat}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: 0}, waited_tokens: [], implied_tokens: [{<escaped>, 0, flat}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_escaped_waits_2(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  scf.if %arg1 {
    %c0_i32_mig7 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig7) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  }
  // Catch escaped token with a full wait.
  amdgcn.wait vm_cnt 0 lgkm_cnt 0
  return
}

// CHECK-LABEL: test_if_flow_1
// CHECK:       Op: func.func @test_if_flow_1(%[[R_0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[R_1:.*]]: i1) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: %[[R_2:.*]] = scf.if %[[R_1]] -> (!amdgcn.read_token<flat>) {...} else {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_3:.*]], %[[R_4:.*]] = amdgcn.global_load_dword dest %[[R_5:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: scf.yield %[[R_4]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_3]], %[[R_4]] = amdgcn.global_load_dword dest %[[R_6:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: scf.yield %[[R_4]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait deps %[[R_2]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 0, flat}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 0, flat}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_if_flow_1(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  // Check waits in both branches with no escaped tokens.
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %c0_i32_mig8 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig8) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  } else {
    %c0_i32_mig9 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %1 addr %arg0 offset c(%c0_i32_mig9) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL: test_if_flow_2
// CHECK:       Op: func.func @test_if_flow_2(%[[R_0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[R_1:.*]]: i1) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: %[[R_2:.*]] = scf.if %[[R_1]] -> (!amdgcn.read_token<flat>) {...} else {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}, {<escaped>, 1, flat}]
// CHECK:       Op: %[[R_3:.*]], %[[R_4:.*]] = amdgcn.global_load_dword dest %[[R_5:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_6:.*]], %[[R_7:.*]] = amdgcn.global_load_dword dest %[[R_5]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_8:.*]], %[[R_9:.*]] = amdgcn.global_load_dword dest %[[R_5]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: scf.yield %[[R_9]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_3]], %[[R_4]] = amdgcn.global_load_dword dest %[[R_10:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: scf.yield %[[R_4]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait deps %[[R_2]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}, {<escaped>, 1, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 0, flat}, {<escaped>, 1, flat}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 0, flat}, {<escaped>, 1, flat}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_if_flow_2(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  // Test unbalanced loads in branches 1.
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %c0_i32_mig10 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig10) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig11 = arith.constant 0 : i32
    %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig11) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig12 = arith.constant 0 : i32
    %result_2, %token_3 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig12) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token_3 : !amdgcn.read_token<flat>
  } else {
    %c0_i32_mig13 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %1 addr %arg0 offset c(%c0_i32_mig13) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL: test_if_flow_3
// CHECK:       Op: func.func @test_if_flow_3(%[[R_0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[R_1:.*]]: i1) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: %[[R_2:.*]] = scf.if %[[R_1]] -> (!amdgcn.read_token<flat>) {...} else {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 1, flat}, {<escaped>, 0, flat}]
// CHECK:       Op: %[[R_3:.*]], %[[R_4:.*]] = amdgcn.global_load_dword dest %[[R_5:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_6:.*]], %[[R_7:.*]] = amdgcn.global_load_dword dest %[[R_5]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_8:.*]], %[[R_9:.*]] = amdgcn.global_load_dword dest %[[R_5]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: scf.yield %[[R_4]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 2, flat}, {%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_3]], %[[R_4]] = amdgcn.global_load_dword dest %[[R_10:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_6]], %[[R_7]] = amdgcn.global_load_dword dest %[[R_5]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: scf.yield %[[R_4]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait deps %[[R_2]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 1, flat}, {<escaped>, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{<escaped>, 0, flat}], wait information = {counts: {vm_cnt: 1, lgkm_cnt: nowait}, waited_tokens: [{%[[R_2]], {{[0-9]*}}, 1, flat}], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 1, flat}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{<escaped>, 0, flat}], wait information = {counts: {vm_cnt: 1, lgkm_cnt: nowait}, waited_tokens: [{%[[R_2]], {{[0-9]*}}, 1, flat}], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 1, flat}]}
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{<escaped>, 0, flat}]
func.func @test_if_flow_3(%arg0: !amdgcn.vgpr<[? + 2]>, %arg1: i1) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  // Test unbalanced loads in branches 2.
  %2 = scf.if %arg1 -> (!amdgcn.read_token<flat>) {
    %c0_i32_mig14 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig14) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig15 = arith.constant 0 : i32
    %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig15) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig16 = arith.constant 0 : i32
    %result_2, %token_3 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig16) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  } else {
    %c0_i32_mig17 = arith.constant 0 : i32
    %result, %token = amdgcn.global_load_dword dest %1 addr %arg0 offset c(%c0_i32_mig17) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    %c0_i32_mig18 = arith.constant 0 : i32
    %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig18) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
    scf.yield %token : !amdgcn.read_token<flat>
  }
  amdgcn.wait deps %2 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL: test_passthrough_pattern
// CHECK:       Op: func.func @test_passthrough_pattern(%[[R_0:.*]]: !amdgcn.vgpr<[? + 2]>) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: %[[R_1:.*]], %[[R_2:.*]] = amdgcn.global_load_dword dest %[[R_3:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_4:.*]], %[[R_5:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 1, flat}, {%[[R_5]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_6:.*]], %[[R_7:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 1, flat}, {%[[R_5]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait deps %[[R_2]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: 2, lgkm_cnt: nowait}, waited_tokens: [{%[[R_2]], {{[0-9]*}}, 2, flat}], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 2, flat}]}
// CHECK:       Op: amdgcn.wait deps %[[R_7]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: 2, lgkm_cnt: nowait}, waited_tokens: [{%[[R_2]], {{[0-9]*}}, 2, flat}], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 2, flat}]}
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {vm_cnt: 0, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_passthrough_pattern(%arg0: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32_mig19 = arith.constant 0 : i32
  %result, %token = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig19) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig20 = arith.constant 0 : i32
  %result_0, %token_1 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig20) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig21 = arith.constant 0 : i32
  %result_2, %token_3 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig21) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  // Test the first wait passes through loads.
  amdgcn.wait deps %token : !amdgcn.read_token<flat>
  amdgcn.wait deps %token_3 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL: test_mixed_smem_dsmem
// CHECK:       Op: func.func @test_mixed_smem_dsmem(%[[R_0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[R_1:.*]]: !amdgcn.vgpr, %[[R_2:.*]]: !amdgcn.vgpr<[? + 2]>) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: %[[R_3:.*]], %[[R_4:.*]] = amdgcn.s_load_dword dest %[[R_5:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<constant>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, constant}]
// CHECK:       Op: %[[R_6:.*]], %[[R_7:.*]] = amdgcn.ds_read_b32 dest %[[R_8:.*]] addr %[[R_1]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 0, constant}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, constant}, {%[[R_7]], {{[0-9]*}}, 0, shared}]
// CHECK:       Op: %[[R_9:.*]], %[[R_10:.*]] = amdgcn.global_load_dword dest %[[R_8]] addr %[[R_2]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_4]], {{[0-9]*}}, 1, constant}, {%[[R_7]], {{[0-9]*}}, 0, shared}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_10]], {{[0-9]*}}, 0, flat}, {%[[R_4]], {{[0-9]*}}, 1, constant}, {%[[R_7]], {{[0-9]*}}, 0, shared}]
// CHECK:       Op: amdgcn.wait deps %[[R_4]] : !amdgcn.read_token<constant>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_10]], {{[0-9]*}}, 0, flat}, {%[[R_4]], {{[0-9]*}}, 1, constant}, {%[[R_7]], {{[0-9]*}}, 0, shared}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_10]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: nowait, lgkm_cnt: 0}, waited_tokens: [], implied_tokens: [{%[[R_4]], {{[0-9]*}}, 1, constant}, {%[[R_7]], {{[0-9]*}}, 0, shared}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_10]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: nowait, lgkm_cnt: 0}, waited_tokens: [], implied_tokens: [{%[[R_4]], {{[0-9]*}}, 1, constant}, {%[[R_7]], {{[0-9]*}}, 0, shared}]}
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_10]], {{[0-9]*}}, 0, flat}]
func.func @test_mixed_smem_dsmem(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.sgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // Test mixed SMEM and DSMEM loads. For CDNA3+, the only safe wait is to wait for everything.
  %c0_i32_mig1 = arith.constant 0 : i32
  %result, %token = amdgcn.s_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig1) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<constant>
  %result_0, %token_1 = amdgcn.ds_read_b32 dest %1 addr %arg1 offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %c0_i32_mig22 = arith.constant 0 : i32
  %result_2, %token_3 = amdgcn.global_load_dword dest %1 addr %arg2 offset c(%c0_i32_mig22) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait deps %token : !amdgcn.read_token<constant>
  return
}

// CHECK-LABEL: test_counts_strength
// CHECK:       Op: func.func @test_counts_strength(%[[R_0:.*]]: !amdgcn.vgpr<[? + 2]>) {...}
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: <Empty>
// CHECK:       Op: %[[R_1:.*]], %[[R_2:.*]] = amdgcn.global_load_dword dest %[[R_3:.*]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_4:.*]], %[[R_5:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 1, flat}, {%[[R_5]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_6:.*]], %[[R_7:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 1, flat}, {%[[R_5]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait vm_cnt 1 deps %[[R_2]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}, {%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: 1, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}]}
// CHECK:       Op: %{{.*}} = arith.constant 0 : i32
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: 1, lgkm_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[R_2]], {{[0-9]*}}, 2, flat}, {%[[R_5]], {{[0-9]*}}, 1, flat}]}
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_8:.*]], %[[R_9:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %{{.*}} = arith.constant 0 : i32
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_10:.*]], %[[R_11:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 1, flat}, {%[[R_9]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 2, flat}, {%[[R_9]], {{[0-9]*}}, 1, flat}, {%[[R_11]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %{{.*}} = arith.constant 0 : i32
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 2, flat}, {%[[R_9]], {{[0-9]*}}, 1, flat}, {%[[R_11]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 2, flat}, {%[[R_9]], {{[0-9]*}}, 1, flat}, {%[[R_11]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: %[[R_12:.*]], %[[R_13:.*]] = amdgcn.global_load_dword dest %[[R_3]] addr %[[R_0]] offset c(%{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 2, flat}, {%[[R_9]], {{[0-9]*}}, 1, flat}, {%[[R_11]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 3, flat}, {%[[R_9]], {{[0-9]*}}, 2, flat}, {%[[R_11]], {{[0-9]*}}, 1, flat}, {%[[R_13]], {{[0-9]*}}, 0, flat}]
// CHECK:       Op: amdgcn.wait vm_cnt 4 deps %[[R_9]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_7]], {{[0-9]*}}, 3, flat}, {%[[R_9]], {{[0-9]*}}, 2, flat}, {%[[R_11]], {{[0-9]*}}, 1, flat}, {%[[R_13]], {{[0-9]*}}, 0, flat}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_11]], {{[0-9]*}}, 1, flat}, {%[[R_13]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: 2, lgkm_cnt: nowait}, waited_tokens: [{%[[R_9]], {{[0-9]*}}, 2, flat}], implied_tokens: [{%[[R_7]], {{[0-9]*}}, 3, flat}, {%[[R_9]], {{[0-9]*}}, 2, flat}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[R_11]], {{[0-9]*}}, 1, flat}, {%[[R_13]], {{[0-9]*}}, 0, flat}], wait information = {counts: {vm_cnt: 2, lgkm_cnt: nowait}, waited_tokens: [{%[[R_9]], {{[0-9]*}}, 2, flat}], implied_tokens: [{%[[R_7]], {{[0-9]*}}, 3, flat}, {%[[R_9]], {{[0-9]*}}, 2, flat}]}
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[R_11]], {{[0-9]*}}, 1, flat}, {%[[R_13]], {{[0-9]*}}, 0, flat}]
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
  // Test that %token is not waited on as vm_cnt is stronger.
  amdgcn.wait vm_cnt 1 deps %token : !amdgcn.read_token<flat>
  %c0_i32_mig26 = arith.constant 0 : i32
  %result_4, %token_5 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig26) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig27 = arith.constant 0 : i32
  %result_6, %token_7 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig27) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %c0_i32_mig28 = arith.constant 0 : i32
  %result_8, %token_9 = amdgcn.global_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig28) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  // Test the token has precedence over vm_cnt.
  amdgcn.wait vm_cnt 4 deps %token_5 : !amdgcn.read_token<flat>
  return
}

// CHECK-LABEL: Symbol: test_smem_zero
// CHECK: amdgcn.wait deps
// CHECK-NEXT: WAIT STATE BEFORE: unhandled tokens = [{%{{.*}}, {{.*}}, 1, constant}, {%{{.*}}, {{.*}}, 0, constant}]
// CHECK-NEXT: WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {vm_cnt: nowait, lgkm_cnt: 0}
func.func @test_smem_zero(%arg0: !amdgcn.sgpr<[? + 2]>, %arg1: !amdgcn.vgpr, %arg2: !amdgcn.vgpr<[? + 2]>) {
  %0 = amdgcn.alloca : !amdgcn.sgpr
  %1 = amdgcn.alloca : !amdgcn.vgpr
  // If there are any SMEM loads, flush them.
  %c0_i32_mig2 = arith.constant 0 : i32
  %dest_res, %token = amdgcn.s_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig2) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<constant>
  %c0_i32_mig3 = arith.constant 0 : i32
  %dest_res1, %token_1 = amdgcn.s_load_dword dest %0 addr %arg0 offset c(%c0_i32_mig3) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<constant>
  amdgcn.wait deps %token : !amdgcn.read_token<constant>
  return
}
