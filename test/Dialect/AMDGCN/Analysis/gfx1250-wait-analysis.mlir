// RUN: aster-opt %s --test-wait-analysis | FileCheck %s


amdgcn.module @gfx1250_tests target = #amdgcn.target<gfx1250> {

// CHECK-LABEL: test_tensor_load_wait
// CHECK:       Op: %[[TOK:.*]] = amdgcn.tensor_load_to_lds
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, tensor}]
// CHECK:       Op: amdgcn.wait_gfx1250 deps %[[TOK]] : !amdgcn.read_token<tensor>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, tensor}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: 0}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, tensor}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: 0}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, tensor}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_tensor_load_wait() {
  %d0 = lsir.alloca : !amdgcn.sgpr<[0 : 4]>
  %d1 = lsir.alloca : !amdgcn.sgpr<[8 : 16]>
  %d2 = lsir.alloca : !amdgcn.sgpr<[16 : 20]>
  %d3 = lsir.alloca : !amdgcn.sgpr<[20 : 24]>
  %tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
      : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<[8 : 16]>, !amdgcn.sgpr<[16 : 20]>, !amdgcn.sgpr<[20 : 24]>) -> !amdgcn.read_token<tensor>
  amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<tensor>
  return
}

// CHECK-LABEL: test_ds_load_wait
// CHECK:       Op: %[[TOK:.*]] = amdgcn.ds_load_b128
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       Op: amdgcn.wait_gfx1250 deps %[[TOK]] : !amdgcn.read_token<shared>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: 0, km_cnt: nowait, tensor_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, ds}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: 0, km_cnt: nowait, tensor_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, ds}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_ds_load_wait(%addr: !amdgcn.vgpr) {
  %dst = lsir.alloca : !amdgcn.vgpr<[0 : 4]>
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.ds_load_b128 dest %dst addr %addr offset c(%c0)
      : outs(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<shared>
  return
}

// CHECK-LABEL: test_global_load_wait
// CHECK:       Op: %[[R:.*]], %[[TOK:.*]] = amdgcn.global_load_dword
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, load}]
// CHECK:       Op: amdgcn.wait_gfx1250 deps %[[TOK]] : !amdgcn.read_token<flat>
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, load}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {load_cnt: 0, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, load}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {load_cnt: 0, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, load}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_global_load_wait(%addr: !amdgcn.vgpr<[? + 2]>, %dst: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %res, %tok = amdgcn.global_load_dword dest %dst addr %addr offset c(%c0)
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<flat>
  return
}

}
