// RUN: aster-opt %s --amdgcn-attach-scheduler=path=%aster_src/test/Dialect/AMDGCN/Transforms/yaml-sched.yaml --aster-apply-sched | FileCheck %s

// CHECK-LABEL:   func.func @test() {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_add_u32 outs %[[ALLOCA_2]] ins %[[ALLOCA_1]], %[[ALLOCA_0]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_3:.*]] = lsir.alloca : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ALLOCA_2]] addr %[[ALLOCA_3]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ALLOCA_2]] addr %[[ALLOCA_3]] : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @test() {
  %c1_i32 = arith.constant 1 : i32
  %0 = amdgcn.alloca : !amdgcn.vgpr
  %1 = lsir.alloca : !amdgcn.vgpr<[? + 2]>
  %2 = amdgcn.alloca : !amdgcn.vgpr
  %3 = amdgcn.alloca : !amdgcn.vgpr
  %vdst0_res = amdgcn.vop2 v_add_u32 outs %3 ins %2, %0 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  %c0_i32 = arith.constant 0 : i32
  %4 = amdgcn.store global_store_dword data %3 addr %1 : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
  %dest_res, %token = amdgcn.load global_load_dword dest %3 addr %1 : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  return
}
