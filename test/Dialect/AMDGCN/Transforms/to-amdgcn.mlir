// RUN: aster-opt %s -aster-to-amdgcn | FileCheck %s

// CHECK-LABEL:   func.func @test_add_i16(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VOP_0:.*]] = amdgcn.vop.add v_add_u16 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VOP_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_add_i16(%dst: !pir.reg<i16 : !amdgcn.vgpr>, %lhs: !pir.reg<i16 : !amdgcn.vgpr>, %rhs: !pir.reg<i16 : !amdgcn.vgpr>) -> !pir.reg<i16 : !amdgcn.vgpr>{
  %res = pir.addi %dst, %lhs, %rhs : !pir.reg<i16 : !amdgcn.vgpr>, !pir.reg<i16 : !amdgcn.vgpr>, !pir.reg<i16 : !amdgcn.vgpr>
  return %res : !pir.reg<i16 : !amdgcn.vgpr>
}

// CHECK-LABEL:   func.func @test_add_i32(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,  %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VOP_0:.*]] = amdgcn.vop.add v_add_u32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr
// CHECK:           return %[[VOP_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_add_i32(%dst: !pir.reg<i32: !amdgcn.vgpr>, %lhs: !pir.reg<i32: !amdgcn.vgpr>, %rhs: !pir.reg<i32: !amdgcn.sgpr>) -> !pir.reg<i32: !amdgcn.vgpr>{
  %res = pir.addi %dst, %lhs, %rhs : !pir.reg<i32: !amdgcn.vgpr>, !pir.reg<i32: !amdgcn.vgpr>, !pir.reg<i32: !amdgcn.sgpr>
  return %res : !pir.reg<i32: !amdgcn.vgpr>
}

// CHECK-LABEL:   func.func @test_add_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr_range<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr_range<[? + 2]>, %[[ARG2:.*]]: !amdgcn.vgpr_range<[? + 2]>) -> !amdgcn.vgpr_range<[? + 2]> {
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.vgpr_range<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[ARG1]] : !amdgcn.vgpr_range<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG2]] : !amdgcn.vgpr_range<[? + 2]>
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]], %[[VOP_0:.*]] = amdgcn.vop.add v_add_co_u32 outs %[[SPLIT_REGISTER_RANGE_0]]#0 carry_out = %[[MAKE_REGISTER_RANGE_0]] ins %[[SPLIT_REGISTER_RANGE_1]]#0, %[[SPLIT_REGISTER_RANGE_2]]#0 : !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]], %[[VOP_1:.*]] = amdgcn.vop.add v_addc_co_u32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 carry_out = %[[MAKE_REGISTER_RANGE_1]] ins %[[SPLIT_REGISTER_RANGE_1]]#1, %[[SPLIT_REGISTER_RANGE_2]]#1 carry_in = %[[VOP_0]] : !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_2]] : !amdgcn.vgpr_range<[? + 2]>
// CHECK:         }
func.func @test_add_i64(%dst: !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>, %lhs: !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>, %rhs: !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>) -> !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>{
  %res = pir.addi %dst, %lhs, %rhs : !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>, !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>, !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>
  return %res : !pir.reg<i64 : !amdgcn.vgpr_range<[? + 2]>>
}
