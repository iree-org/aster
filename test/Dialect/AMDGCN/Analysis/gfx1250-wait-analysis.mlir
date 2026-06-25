// RUN: aster-opt %s --test-wait-analysis | FileCheck %s


amdgcn.module @gfx1250_tests target = #amdgcn.target<gfx1250> {

// CHECK-LABEL: test_tensor_load_wait
// CHECK:       Op: %[[TOK:.*]] = amdgcn.tensor_load_to_lds
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, tensor}]
// CHECK:       Op: %{{.*}} = amdgcn.wait_gfx1250 deps %[[TOK]] : !amdgcn.read_token<tensor> -> !amdgcn.fence_token
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, tensor}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: 0, async_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, tensor}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: 0, async_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, tensor}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_tensor_load_wait() {
  %d0 = lsir.alloca : !amdgcn.sgpr<[0 : 4]>
  %d1 = lsir.alloca : !amdgcn.sgpr<[8 : 16]>
  %d2 = lsir.alloca : !amdgcn.sgpr<[16 : 20]>
  %d3 = lsir.alloca : !amdgcn.sgpr<[20 : 24]>
  %tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
      : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<[8 : 16]>, !amdgcn.sgpr<[16 : 20]>, !amdgcn.sgpr<[20 : 24]>) -> !amdgcn.read_token<tensor>
  %wf0 = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<tensor> -> !amdgcn.fence_token
  return
}

// CHECK-LABEL: test_ds_load_wait
// CHECK:       Op: %[[TOK:.*]] = amdgcn.ds_load_b128
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       Op: %{{.*}} = amdgcn.wait_gfx1250 deps %[[TOK]] : !amdgcn.read_token<shared> -> !amdgcn.fence_token
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: 0, km_cnt: nowait, tensor_cnt: nowait, async_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, ds}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: 0, km_cnt: nowait, tensor_cnt: nowait, async_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, ds}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_ds_load_wait(%addr: !amdgcn.vgpr) {
  %dst = lsir.alloca : !amdgcn.vgpr<[0 : 4]>
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.ds_load_b128 dest %dst addr %addr offset c(%c0)
      : outs(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %wf1 = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<shared> -> !amdgcn.fence_token
  return
}

// CHECK-LABEL: test_global_load_wait
// CHECK:       Op: %[[R:.*]], %[[TOK:.*]] = amdgcn.global_load_dword
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, load}]
// CHECK:       Op: %{{.*}} = amdgcn.wait_gfx1250 deps %[[TOK]] : !amdgcn.read_token<flat> -> !amdgcn.fence_token
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, load}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {load_cnt: 0, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: nowait, async_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, load}]}
// CHECK:       Op: func.return
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [], wait information = {counts: {load_cnt: 0, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: nowait, async_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, load}]}
// CHECK:       	WAIT STATE AFTER: <Empty>
func.func @test_global_load_wait(%addr: !amdgcn.vgpr<[? + 2]>, %dst: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %res, %tok = amdgcn.global_load_dword dest %dst addr %addr offset c(%c0)
      : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<flat>
  %wf2 = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<flat> -> !amdgcn.fence_token
  return
}

// CHECK-LABEL: test_km_wait_does_not_drain_ds
// CHECK:       Op: %[[DS_TOK:.*]] = amdgcn.ds_load_b128
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       Op: %{{.*}}, %[[KM_TOK:.*]] = amdgcn.s_load_dword
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[KM_TOK]], {{[0-9]*}}, 0, scalar_read}, {%[[DS_TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       Op: %{{.*}} = amdgcn.wait_gfx1250 deps %[[KM_TOK]] : !amdgcn.read_token<constant> -> !amdgcn.fence_token
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[KM_TOK]], {{[0-9]*}}, 0, scalar_read}, {%[[DS_TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: 0, tensor_cnt: nowait, async_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[KM_TOK]], {{[0-9]*}}, 0, scalar_read}]}
func.func @test_km_wait_does_not_drain_ds(%ds_addr: !amdgcn.vgpr, %km_addr: !amdgcn.sgpr<[? + 2]>) {
  %ds_dst = lsir.alloca : !amdgcn.vgpr<[0 : 4]>
  %km_dst = amdgcn.alloca : !amdgcn.sgpr
  %c0_i32_mig29 = arith.constant 0 : i32
  %ds_tok = amdgcn.ds_load_b128 dest %ds_dst addr %ds_addr offset c(%c0_i32_mig29) : outs(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %km_r, %km_tok = amdgcn.s_load_dword dest %km_dst addr %km_addr offset c(%c0_i32_mig29) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) mods(i32) -> !amdgcn.read_token<constant>
  // On gfx1250 km and ds are independent counters: a km wait must not drain ds.
  %wf3 = amdgcn.wait_gfx1250 deps %km_tok : !amdgcn.read_token<constant> -> !amdgcn.fence_token
  return
}

// CHECK-LABEL: test_tensor_wait_does_not_drain_ds
// CHECK:       Op: %[[DS_TOK:.*]] = amdgcn.ds_load_b128
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       Op: %[[TEN_TOK:.*]] = amdgcn.tensor_load_to_lds
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}, {%[[TEN_TOK]], {{[0-9]*}}, 0, tensor}]
// CHECK:       Op: %{{.*}} = amdgcn.wait_gfx1250 deps %[[TEN_TOK]] : !amdgcn.read_token<tensor> -> !amdgcn.fence_token
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}, {%[[TEN_TOK]], {{[0-9]*}}, 0, tensor}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: 0, async_cnt: nowait}, waited_tokens: [], implied_tokens: [{%[[TEN_TOK]], {{[0-9]*}}, 0, tensor}]}
func.func @test_tensor_wait_does_not_drain_ds(%ds_addr: !amdgcn.vgpr) {
  %ds_dst = lsir.alloca : !amdgcn.vgpr<[0 : 4]>
  %d0 = lsir.alloca : !amdgcn.sgpr<[0 : 4]>
  %d1 = lsir.alloca : !amdgcn.sgpr<[8 : 16]>
  %d2 = lsir.alloca : !amdgcn.sgpr<[16 : 20]>
  %d3 = lsir.alloca : !amdgcn.sgpr<[20 : 24]>
  %c0_i32_mig30 = arith.constant 0 : i32
  %ds_tok = amdgcn.ds_load_b128 dest %ds_dst addr %ds_addr offset c(%c0_i32_mig30) : outs(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %tensor_tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3 : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<[8 : 16]>, !amdgcn.sgpr<[16 : 20]>, !amdgcn.sgpr<[20 : 24]>) -> !amdgcn.read_token<tensor>
  // On gfx1250 tensor and ds are independent counters: a tensor wait must not drain ds.
  %wf4 = amdgcn.wait_gfx1250 deps %tensor_tok : !amdgcn.read_token<tensor> -> !amdgcn.fence_token
  return
}

// CHECK-LABEL: test_async_cluster_load_wait
// CHECK:       Op: %[[TOK:.*]] = amdgcn.cluster_load_async_to_lds_b32
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, async}]
// CHECK:       Op: %{{.*}} = amdgcn.wait_gfx1250 deps %[[TOK]] : !amdgcn.read_token<async> -> !amdgcn.fence_token
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, async}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: nowait, async_cnt: 0}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, async}]}
func.func @test_async_cluster_load_wait(%lds_dst: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>, %m0: !amdgcn.m0<0>) {
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.cluster_load_async_to_lds_b32 lds_dest %lds_dst addr %addr m0 %m0 offset c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, !amdgcn.m0<0>) mods(i32) -> !amdgcn.read_token<async>
  %wf = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<async> -> !amdgcn.fence_token
  return
}

// CHECK-LABEL: test_global_load_async_to_lds_wait
// CHECK:       Op: %[[TOK:.*]] = amdgcn.global_load_async_to_lds_b32
// CHECK:       	WAIT STATE BEFORE: <Empty>
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, async}]
// CHECK:       Op: %{{.*}} = amdgcn.wait_gfx1250 deps %[[TOK]] : !amdgcn.read_token<async> -> !amdgcn.fence_token
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[TOK]], {{[0-9]*}}, 0, async}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: nowait, async_cnt: 0}, waited_tokens: [], implied_tokens: [{%[[TOK]], {{[0-9]*}}, 0, async}]}
func.func @test_global_load_async_to_lds_wait(%addr: !amdgcn.vgpr<[? + 2]>, %lds_addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %tok = amdgcn.global_load_async_to_lds_b32 addr %addr lds_addr %lds_addr offset c(%c0)
      : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<async>
  %wf = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<async> -> !amdgcn.fence_token
  return
}

// CHECK-LABEL: test_async_wait_does_not_drain_ds
// CHECK:       Op: %[[DS_TOK:.*]] = amdgcn.ds_load_b128
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}]
// CHECK:       Op: %[[ASYNC_TOK:.*]] = amdgcn.global_load_async_to_lds_b32
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}, {%[[ASYNC_TOK]], {{[0-9]*}}, 0, async}]
// CHECK:       Op: %{{.*}} = amdgcn.wait_gfx1250 deps %[[ASYNC_TOK]] : !amdgcn.read_token<async> -> !amdgcn.fence_token
// CHECK:       	WAIT STATE BEFORE: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}, {%[[ASYNC_TOK]], {{[0-9]*}}, 0, async}]
// CHECK:       	WAIT STATE AFTER: unhandled tokens = [{%[[DS_TOK]], {{[0-9]*}}, 0, ds}], wait information = {counts: {load_cnt: nowait, store_cnt: nowait, ds_cnt: nowait, km_cnt: nowait, tensor_cnt: nowait, async_cnt: 0}, waited_tokens: [], implied_tokens: [{%[[ASYNC_TOK]], {{[0-9]*}}, 0, async}]}
func.func @test_async_wait_does_not_drain_ds(%ds_addr: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>, %lds_addr: !amdgcn.vgpr) {
  %ds_dst = lsir.alloca : !amdgcn.vgpr<[0 : 4]>
  %c0 = arith.constant 0 : i32
  %ds_tok = amdgcn.ds_load_b128 dest %ds_dst addr %ds_addr offset c(%c0) : outs(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
  %async_tok = amdgcn.global_load_async_to_lds_b32 addr %addr lds_addr %lds_addr offset c(%c0) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<async>
  // On gfx1250 async and ds are independent counters: an async wait must not drain ds.
  %wf = amdgcn.wait_gfx1250 deps %async_tok : !amdgcn.read_token<async> -> !amdgcn.fence_token
  return
}

}
