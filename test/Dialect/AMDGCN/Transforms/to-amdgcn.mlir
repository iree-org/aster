// RUN: aster-opt %s -aster-to-amdgcn --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @test_assume_noalias(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) {
// CHECK:           return %[[ARG0]], %[[ARG1]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_assume_noalias(%ptr1: !amdgcn.sgpr<[? + 2]>, %ptr2: !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) {
  %ptr1_noalias, %ptr2_noalias = lsir.assume_noalias %ptr1, %ptr2
    : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>)
  return %ptr1_noalias, %ptr2_noalias : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_add_i32(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_add_u32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_add_i32(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.addi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_add_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_lshl_add_u64 outs %[[ARG0]] ins %[[ARG1]], %[[CONSTANT_0]] src2 = %[[ARG2]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, i32, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_add_i64(%dst: !amdgcn.vgpr<[? + 2]>, %lhs: !amdgcn.vgpr<[? + 2]>, %rhs: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>{
  %res = lsir.addi i64 %dst, %lhs, %rhs : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
  return %res : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_sadd_i32(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_add_u32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_sadd_i32(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.addi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_sadd_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[ARG1]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG2]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_add_u32 outs %[[SPLIT_REGISTER_RANGE_0]]#0 ins %[[SPLIT_REGISTER_RANGE_1]]#0, %[[SPLIT_REGISTER_RANGE_2]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_addc_u32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[SPLIT_REGISTER_RANGE_1]]#1, %[[SPLIT_REGISTER_RANGE_2]]#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_sadd_i64(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.addi i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_sub_i32(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_sub_u32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_sub_i32(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.subi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_sub_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[ARG1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG2]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]], %[[VAL_1:.*]] = amdgcn.vop3 v_sub_co_u32_e64 outs %[[SPLIT_REGISTER_RANGE_0]]#0 dst1 = %[[MAKE_REGISTER_RANGE_0]] ins %[[SPLIT_REGISTER_RANGE_1]]#0, %[[SPLIT_REGISTER_RANGE_2]]#0 : !amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = amdgcn.vop3 v_subb_co_u32_e64 outs %[[SPLIT_REGISTER_RANGE_0]]#1 dst1 = %[[MAKE_REGISTER_RANGE_0]] ins %[[SPLIT_REGISTER_RANGE_1]]#1, %[[SPLIT_REGISTER_RANGE_2]]#1 src2 = %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[VAL_0]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_sub_i64(%dst: !amdgcn.vgpr<[? + 2]>, %lhs: !amdgcn.vgpr<[? + 2]>, %rhs: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>{
  %res = lsir.subi i64 %dst, %lhs, %rhs : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
  return %res : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_ssub_i32(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_sub_u32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_ssub_i32(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.subi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_ssub_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[ARG1]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG2]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_sub_u32 outs %[[SPLIT_REGISTER_RANGE_0]]#0 ins %[[SPLIT_REGISTER_RANGE_1]]#0, %[[SPLIT_REGISTER_RANGE_2]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_subb_u32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[SPLIT_REGISTER_RANGE_1]]#1, %[[SPLIT_REGISTER_RANGE_2]]#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_ssub_i64(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.subi i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_mul_i32(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_mul_lo_u32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_mul_i32(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.muli i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_mul_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[ARG2]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_mul_lo_u32 outs %[[ALLOCA_0]] ins %[[SPLIT_REGISTER_RANGE_1]]#1, %[[SPLIT_REGISTER_RANGE_0]]#0 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop3 v_mul_lo_u32 outs %[[ALLOCA_1]] ins %[[SPLIT_REGISTER_RANGE_1]]#0, %[[SPLIT_REGISTER_RANGE_0]]#1 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = amdgcn.vop3 v_mad_u64_u32 outs %[[ARG0]] dst1 = %[[MAKE_REGISTER_RANGE_0]] ins %[[SPLIT_REGISTER_RANGE_1]]#0, %[[SPLIT_REGISTER_RANGE_0]]#0 src2 = %[[CONSTANT_0]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr, i32
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[VAL_2]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[VAL_4:.*]] = amdgcn.vop3 v_add3_u32 outs %[[SPLIT_REGISTER_RANGE_2]]#1 ins %[[SPLIT_REGISTER_RANGE_2]]#1, %[[VAL_1]] src2 = %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[SPLIT_REGISTER_RANGE_2]]#0, %[[VAL_4]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_mul_i64(%dst: !amdgcn.vgpr<[? + 2]>, %lhs: !amdgcn.vgpr<[? + 2]>, %rhs: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>{
  %res = lsir.muli i64 %dst, %lhs, %rhs : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
  return %res : !amdgcn.vgpr<[? + 2]>
}


// CHECK-LABEL:   func.func @test_smul_i32(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_mul_i32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_smul_i32(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.muli i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_smul_i64(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[ARG1]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG2]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_mul_i32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[SPLIT_REGISTER_RANGE_2]]#0, %[[SPLIT_REGISTER_RANGE_1]]#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_mul_hi_u32 outs %[[ALLOCA_0]] ins %[[SPLIT_REGISTER_RANGE_2]]#0, %[[SPLIT_REGISTER_RANGE_1]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_2:.*]] = amdgcn.sop2 s_mul_i32 outs %[[ALLOCA_1]] ins %[[SPLIT_REGISTER_RANGE_2]]#1, %[[SPLIT_REGISTER_RANGE_1]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_3:.*]] = amdgcn.sop2 s_add_i32 outs %[[VAL_0]] ins %[[VAL_1]], %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_4:.*]] = amdgcn.sop2 s_mul_i32 outs %[[SPLIT_REGISTER_RANGE_0]]#0 ins %[[SPLIT_REGISTER_RANGE_2]]#0, %[[SPLIT_REGISTER_RANGE_1]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_5:.*]] = amdgcn.sop2 s_add_i32 outs %[[VAL_3]] ins %[[VAL_3]], %[[VAL_2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[VAL_4]], %[[VAL_5]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_smul_i64(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.muli i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_shl_i16_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_lshlrev_b16 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_shl_i16_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.shli i16 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_shl_i32_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_lshlrev_b32_e32 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_shl_i32_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.shli i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_shl_i64_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_lshlrev_b64 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_shl_i64_vgpr(%dst: !amdgcn.vgpr<[? + 2]>, %lhs: !amdgcn.vgpr<[? + 2]>, %rhs: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>{
  %res = lsir.shli i64 %dst, %lhs, %rhs : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
  return %res : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_shl_i32_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_lshl_b32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_shl_i32_sgpr(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.shli i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_shl_i64_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_lshl_b64 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_shl_i64_sgpr(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.shli i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_shrsi_i16_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_ashrrev_i16 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_shrsi_i16_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.shrsi i16 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_shrsi_i32_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_ashrrev_i32 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_shrsi_i32_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.shrsi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_shrsi_i64_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_ashrrev_i64 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_shrsi_i64_vgpr(%dst: !amdgcn.vgpr<[? + 2]>, %lhs: !amdgcn.vgpr<[? + 2]>, %rhs: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>{
  %res = lsir.shrsi i64 %dst, %lhs, %rhs : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
  return %res : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_shrsi_i32_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_ashr_i32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_shrsi_i32_sgpr(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.shrsi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_shrsi_i64_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_ashr_i64 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_shrsi_i64_sgpr(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.shrsi i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_shrui_i16_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_lshrrev_b16 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_shrui_i16_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.shrui i16 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_shrui_i32_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_lshrrev_b32 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_shrui_i32_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.shrui i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_shrui_i64_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_lshrrev_b64 outs %[[ARG0]] ins %[[ARG2]], %[[ARG1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_shrui_i64_vgpr(%dst: !amdgcn.vgpr<[? + 2]>, %lhs: !amdgcn.vgpr<[? + 2]>, %rhs: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]>{
  %res = lsir.shrui i64 %dst, %lhs, %rhs : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>
  return %res : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_shrui_i32_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_lshr_b32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_shrui_i32_sgpr(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.shrui i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_shrui_i64_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_lshr_b64 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_shrui_i64_sgpr(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.shrui i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_and_i32_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_and_b32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_and_i32_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.andi i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_and_i32_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_and_b32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_and_i32_sgpr(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.andi i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_and_i64_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_and_b64 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_and_i64_sgpr(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.andi i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_or_i32_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_or_b32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_or_i32_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.ori i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_or_i32_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_or_b32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_or_i32_sgpr(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.ori i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_or_i64_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_or_b64 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_or_i64_sgpr(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.ori i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_xor_i32_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_xor_b32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_xor_i32_vgpr(%dst: !amdgcn.vgpr, %lhs: !amdgcn.vgpr, %rhs: !amdgcn.vgpr) -> !amdgcn.vgpr{
  %res = lsir.xori i32 %dst, %lhs, %rhs : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_xor_i32_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_xor_b32 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_xor_i32_sgpr(%dst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) -> !amdgcn.sgpr{
  %res = lsir.xori i32 %dst, %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_xor_i64_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_xor_b64 outs %[[ARG0]] ins %[[ARG1]], %[[ARG2]] : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_xor_i64_sgpr(%dst: !amdgcn.sgpr<[? + 2]>, %lhs: !amdgcn.sgpr<[? + 2]>, %rhs: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]>{
  %res = lsir.xori i64 %dst, %lhs, %rhs : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_load_global_dword(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_global_dword(%dst: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_load_global_dword_with_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 16 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_global_dword_with_offset(%dst: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c16 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_load_global_dwordx2(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dwordx2 dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_load_global_dwordx2(%dst: !amdgcn.vgpr<[? + 2]>, %addr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_load_global_dwordx3(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 3]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 3]> {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dwordx3 dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.vgpr<[? + 3]>) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 3]>
// CHECK:         }
func.func @test_load_global_dwordx3(%dst: !amdgcn.vgpr<[? + 3]>, %addr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 3]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 3]>, !amdgcn.vgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.vgpr<[? + 3]>
}

// CHECK-LABEL:   func.func @test_load_global_dwordx4(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 4]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 4]> {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dwordx4 dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 4]>
// CHECK:         }
func.func @test_load_global_dwordx4(%dst: !amdgcn.vgpr<[? + 4]>, %addr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 4]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.vgpr<[? + 4]>
}

// CHECK-LABEL:   func.func @test_load_smem_dword(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dword dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_load_smem_dword(%dst: !amdgcn.sgpr, %addr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_load_smem_dword_with_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 32 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dword dest %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_load_smem_dword_with_offset(%dst: !amdgcn.sgpr, %addr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c32 : !amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_load_smem_dwordx2(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dwordx2 dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.sgpr<[? + 2]>) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_load_smem_dwordx2(%dst: !amdgcn.sgpr<[? + 2]>, %addr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.sgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_load_smem_dwordx4(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 4]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 4]> {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dwordx4 dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.sgpr<[? + 4]>) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 4]>
// CHECK:         }
func.func @test_load_smem_dwordx4(%dst: !amdgcn.sgpr<[? + 4]>, %addr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 4]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.sgpr<[? + 4]>
}

// CHECK-LABEL:   func.func @test_load_smem_dwordx8(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 8]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 8]> {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dwordx8 dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.sgpr<[? + 8]>) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 8]>
// CHECK:         }
func.func @test_load_smem_dwordx8(%dst: !amdgcn.sgpr<[? + 8]>, %addr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 8]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.sgpr<[? + 8]>, !amdgcn.sgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.sgpr<[? + 8]>
}

// CHECK-LABEL:   func.func @test_load_smem_dwordx16(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 16]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 16]> {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dwordx16 dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.sgpr<[? + 16]>) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr<[? + 16]>
// CHECK:         }
func.func @test_load_smem_dwordx16(%dst: !amdgcn.sgpr<[? + 16]>, %addr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 16]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.sgpr<[? + 16]>, !amdgcn.sgpr<[? + 2]>, i32, i32
  return %res : !amdgcn.sgpr<[? + 16]>
}

// CHECK-LABEL:   func.func @test_load_local_dword(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load ds_read_b32 dest %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_local_dword(%dst: !amdgcn.vgpr, %addr: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<local, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_load_local_dword_with_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 64 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load ds_read_b32 dest %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_load_local_dword_with_offset(%dst: !amdgcn.vgpr, %addr: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %res, %token = lsir.load #amdgcn.addr_space<local, read_only> %dst, %addr, %c0, %c64 : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_load_local_dwordx2(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load ds_read_b64 dest %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_load_local_dwordx2(%dst: !amdgcn.vgpr<[? + 2]>, %addr: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<local, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32, i32
  return %res : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @test_load_local_dwordx3(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 3]>, %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 3]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load ds_read_b96 dest %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<[? + 3]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 3]>
// CHECK:         }
func.func @test_load_local_dwordx3(%dst: !amdgcn.vgpr<[? + 3]>, %addr: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 3]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<local, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 3]>, !amdgcn.vgpr, i32, i32
  return %res : !amdgcn.vgpr<[? + 3]>
}

// CHECK-LABEL:   func.func @test_load_local_dwordx4(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 4]>, %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 4]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load ds_read_b128 dest %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr<[? + 4]>
// CHECK:         }
func.func @test_load_local_dwordx4(%dst: !amdgcn.vgpr<[? + 4]>, %addr: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 4]> {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<local, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, i32, i32
  return %res : !amdgcn.vgpr<[? + 4]>
}

// CHECK-LABEL:   func.func @test_store_global_dword(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @test_store_global_dword(%data: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_global_dword_with_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 16 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @test_store_global_dword_with_offset(%data: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c16 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_global_dwordx2(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dwordx2 data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @test_store_global_dwordx2(%data: !amdgcn.vgpr<[? + 2]>, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_global_dwordx3(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 3]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dwordx3 data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<[? + 3]>, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @test_store_global_dwordx3(%data: !amdgcn.vgpr<[? + 3]>, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 3]>, !amdgcn.vgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_global_dwordx4(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 4]>, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dwordx4 data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           return
// CHECK:         }
func.func @test_store_global_dwordx4(%data: !amdgcn.vgpr<[? + 4]>, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_smem_dword(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store s_store_dword data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<constant>
// CHECK:           return
// CHECK:         }
func.func @test_store_smem_dword(%data: !amdgcn.sgpr, %addr: !amdgcn.sgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_smem_dword_with_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 32 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store s_store_dword data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<constant>
// CHECK:           return
// CHECK:         }
func.func @test_store_smem_dword_with_offset(%data: !amdgcn.sgpr, %addr: !amdgcn.sgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c32 : !amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_smem_dwordx2(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store s_store_dwordx2 data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<constant>
// CHECK:           return
// CHECK:         }
func.func @test_store_smem_dwordx2(%data: !amdgcn.sgpr<[? + 2]>, %addr: !amdgcn.sgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_smem_dwordx4(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 4]>, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store s_store_dwordx4 data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.write_token<constant>
// CHECK:           return
// CHECK:         }
func.func @test_store_smem_dwordx4(%data: !amdgcn.sgpr<[? + 4]>, %addr: !amdgcn.sgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_local_dword(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
// CHECK:           return
// CHECK:         }
func.func @test_store_local_dword(%data: !amdgcn.vgpr, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<local, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_local_dword_with_offset(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 64 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
// CHECK:           return
// CHECK:         }
func.func @test_store_local_dword_with_offset(%data: !amdgcn.vgpr, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32
  %token = lsir.store #amdgcn.addr_space<local, read_write> %data, %addr, %c0, %c64 : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_local_dwordx2(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b64 data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
// CHECK:           return
// CHECK:         }
func.func @test_store_local_dwordx2(%data: !amdgcn.vgpr<[? + 2]>, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<local, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_local_dwordx3(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 3]>, %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b96 data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr<[? + 3]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
// CHECK:           return
// CHECK:         }
func.func @test_store_local_dwordx3(%data: !amdgcn.vgpr<[? + 3]>, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<local, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 3]>, !amdgcn.vgpr, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_store_local_dwordx4(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 4]>, %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b128 data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
// CHECK:           return
// CHECK:         }
func.func @test_store_local_dwordx4(%data: !amdgcn.vgpr<[? + 4]>, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<local, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, i32, i32
  return
}

// CHECK-LABEL:   func.func @test_wait_single_vmem_load(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 1
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_wait_single_vmem_load(%dst: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  lsir.wait %token : !lsir.load_token
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_wait_single_vmem_store(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr<[? + 2]>) {
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG0]] addr %[[ARG1]] : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 1
// CHECK:           return
// CHECK:         }
func.func @test_wait_single_vmem_store(%data: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<global, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  lsir.wait %token : !lsir.store_token
  return
}

// CHECK-LABEL:   func.func @test_wait_single_smem_load(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dword dest %[[ARG0]] addr %[[ARG1]] : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 1
// CHECK:           return %[[VAL_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_wait_single_smem_load(%dst: !amdgcn.sgpr, %addr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<global, read_only> %dst, %addr, %c0, %c0 : !amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32, i32
  lsir.wait %token : !lsir.load_token
  return %res : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_wait_single_ds_load(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load ds_read_b32 dest %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 1
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_wait_single_ds_load(%dst: !amdgcn.vgpr, %addr: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %res, %token = lsir.load #amdgcn.addr_space<local, read_only> %dst, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32
  lsir.wait %token : !lsir.load_token
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_wait_single_ds_store(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[STORE_0:.*]] = amdgcn.store ds_write_b32 data %[[ARG0]] addr %[[ARG1]] offset c(%[[CONSTANT_0]]) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 1
// CHECK:           return
// CHECK:         }
func.func @test_wait_single_ds_store(%data: !amdgcn.vgpr, %addr: !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %token = lsir.store #amdgcn.addr_space<local, read_write> %data, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32
  lsir.wait %token : !lsir.store_token
  return
}

// CHECK-LABEL:   func.func @test_wait_multiple_vmem_loads(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG0]] addr %[[ARG2]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load global_load_dword dest %[[ARG1]] addr %[[ARG2]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>, i32) -> !amdgcn.read_token<flat>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 2
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }
func.func @test_wait_multiple_vmem_loads(%dst0: !amdgcn.vgpr, %dst1: !amdgcn.vgpr, %addr: !amdgcn.vgpr<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %res0, %token0 = lsir.load #amdgcn.addr_space<global, read_only> %dst0, %addr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  %res1, %token1 = lsir.load #amdgcn.addr_space<global, read_only> %dst1, %addr, %c0, %c4 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  lsir.wait %token0, %token1 : !lsir.load_token, !lsir.load_token
  return %res0, %res1 : !amdgcn.vgpr, !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_wait_multiple_smem_loads(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.sgpr, !amdgcn.sgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load s_load_dword dest %[[ARG0]] addr %[[ARG2]] : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load s_load_dword dest %[[ARG1]] addr %[[ARG2]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> lgkmcnt = 2
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         }
func.func @test_wait_multiple_smem_loads(%dst0: !amdgcn.sgpr, %dst1: !amdgcn.sgpr, %addr: !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.sgpr, !amdgcn.sgpr) {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %res0, %token0 = lsir.load #amdgcn.addr_space<global, read_only> %dst0, %addr, %c0, %c0 : !amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32, i32
  %res1, %token1 = lsir.load #amdgcn.addr_space<global, read_only> %dst1, %addr, %c0, %c4 : !amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32, i32
  lsir.wait %token0, %token1 : !lsir.load_token, !lsir.load_token
  return %res0, %res1 : !amdgcn.sgpr, !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_wait_mixed_vmem_and_smem(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.sgpr, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG3:.*]]: !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.sgpr) {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG0]] addr %[[ARG2]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load s_load_dword dest %[[ARG1]] addr %[[ARG3]] : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 1 lgkmcnt = 1
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.sgpr
// CHECK:         }
func.func @test_wait_mixed_vmem_and_smem(%vdst: !amdgcn.vgpr, %sdst: !amdgcn.sgpr, %vaddr: !amdgcn.vgpr<[? + 2]>, %saddr: !amdgcn.sgpr<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.sgpr) {
  %c0 = arith.constant 0 : i32
  %vres, %vtoken = lsir.load #amdgcn.addr_space<global, read_only> %vdst, %vaddr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  %sres, %stoken = lsir.load #amdgcn.addr_space<global, read_only> %sdst, %saddr, %c0, %c0 : !amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>, i32, i32
  lsir.wait %vtoken, %stoken : !lsir.load_token, !lsir.load_token
  return %vres, %sres : !amdgcn.vgpr, !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_wait_mixed_vmem_and_ds(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG3:.*]]: !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG0]] addr %[[ARG2]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[VAL_1:.*]], %[[LOAD_1:.*]] = amdgcn.load ds_read_b32 dest %[[ARG1]] addr %[[ARG3]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 1 lgkmcnt = 1
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }
func.func @test_wait_mixed_vmem_and_ds(%vdst: !amdgcn.vgpr, %ddst: !amdgcn.vgpr, %vaddr: !amdgcn.vgpr<[? + 2]>, %daddr: !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %c0 = arith.constant 0 : i32
  %vres, %vtoken = lsir.load #amdgcn.addr_space<global, read_only> %vdst, %vaddr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  %dres, %dtoken = lsir.load #amdgcn.addr_space<local, read_only> %ddst, %daddr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr, i32, i32
  lsir.wait %vtoken, %dtoken : !lsir.load_token, !lsir.load_token
  return %vres, %dres : !amdgcn.vgpr, !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_wait_load_and_store(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.vgpr<[? + 2]>, %[[ARG3:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr {
// CHECK:           %[[VAL_0:.*]], %[[LOAD_0:.*]] = amdgcn.load global_load_dword dest %[[ARG0]] addr %[[ARG2]] : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
// CHECK:           %[[STORE_0:.*]] = amdgcn.store global_store_dword data %[[ARG1]] addr %[[ARG3]] : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
// CHECK:           amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 2
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_wait_load_and_store(%dst: !amdgcn.vgpr, %data: !amdgcn.vgpr, %laddr: !amdgcn.vgpr<[? + 2]>, %saddr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr {
  %c0 = arith.constant 0 : i32
  %res, %ltoken = lsir.load #amdgcn.addr_space<global, read_only> %dst, %laddr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  %stoken = lsir.store #amdgcn.addr_space<global, read_write> %data, %saddr, %c0, %c0 : !amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>, i32, i32
  lsir.wait %ltoken, %stoken : !lsir.load_token, !lsir.store_token
  return %res : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @test_mov_constant_to_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ARG0]], %[[CONSTANT_0]] : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @test_mov_constant_to_vgpr(%dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c42 = arith.constant 42 : i32
  %res = lsir.mov %dst, %c42 : !amdgcn.vgpr, i32
  return %res : !amdgcn.vgpr
}

// Test ExtSIOp (sign extension i32->i64) - SGPR
// CHECK-LABEL:   func.func @test_extsi_i32_to_i64_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 31 : i32
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_ashr_i32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[ARG1]], %[[CONSTANT_0]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_extsi_i32_to_i64_sgpr(%dst: !amdgcn.sgpr<[? + 2]>, %value: !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]> {
  %res = lsir.extsi i64 from i32 %dst, %value : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr
  return %res : !amdgcn.sgpr<[? + 2]>
}

// Test ExtSIOp (sign extension i32->i64) - VGPR
// CHECK-LABEL:   func.func @test_extsi_i32_to_i64_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 31 : i32
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_ashrrev_i32_e64 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[CONSTANT_0]], %[[ARG1]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_extsi_i32_to_i64_vgpr(%dst: !amdgcn.vgpr<[? + 2]>, %value: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %res = lsir.extsi i64 from i32 %dst, %value : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  return %res : !amdgcn.vgpr<[? + 2]>
}

// Test ExtUIOp (zero extension i32->i64) - SGPR
// CHECK-LABEL:   func.func @test_extui_i32_to_i64_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop1 s_mov_b32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[CONSTANT_0]] : !amdgcn.sgpr, i32
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @test_extui_i32_to_i64_sgpr(%dst: !amdgcn.sgpr<[? + 2]>, %value: !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]> {
  %res = lsir.extui i64 from i32 %dst, %value : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr
  return %res : !amdgcn.sgpr<[? + 2]>
}

// Test ExtUIOp (zero extension i32->i64) - VGPR
// CHECK-LABEL:   func.func @test_extui_i32_to_i64_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[SPLIT_REGISTER_RANGE_0]]#1, %[[CONSTANT_0]] : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @test_extui_i32_to_i64_vgpr(%dst: !amdgcn.vgpr<[? + 2]>, %value: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %res = lsir.extui i64 from i32 %dst, %value : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  return %res : !amdgcn.vgpr<[? + 2]>
}

// Trivial: no offsets - ptr_add is replaced with ptr
// CHECK-LABEL:   func.func @ptr_add_trivial_sgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           return %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_trivial_sgpr(%ptr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr : !amdgcn.sgpr<[? + 2]>
  return %0 : !amdgcn.sgpr<[? + 2]>
}

// Trivial: no offsets - VGPR
// CHECK-LABEL:   func.func @ptr_add_trivial_vgpr(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           return %[[ARG0]] : !amdgcn.vgpr
// CHECK:         }
func.func @ptr_add_trivial_vgpr(%ptr: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %ptr : !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

// SGPR ptr + constant offset only
// CHECK-LABEL:   func.func @ptr_add_sgpr_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 31 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 16 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_0]] ins %[[CONSTANT_1]] : !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_ashr_i32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[VAL_0]], %[[CONSTANT_0]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_2]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_3:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[VAL_2:.*]] = amdgcn.sop2 s_add_u32 outs %[[SPLIT_REGISTER_RANGE_1]]#0 ins %[[SPLIT_REGISTER_RANGE_2]]#0, %[[SPLIT_REGISTER_RANGE_3]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_3:.*]] = amdgcn.sop2 s_addc_u32 outs %[[SPLIT_REGISTER_RANGE_1]]#1 ins %[[SPLIT_REGISTER_RANGE_2]]#1, %[[SPLIT_REGISTER_RANGE_3]]#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_3:.*]] = amdgcn.make_register_range %[[VAL_2]], %[[VAL_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_3]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_sgpr_const(%ptr: !amdgcn.sgpr<[? + 2]>) -> !amdgcn.sgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr c_off = 16 : !amdgcn.sgpr<[? + 2]>
  return %0 : !amdgcn.sgpr<[? + 2]>
}

// SGPR ptr + dynamic offset only (VGPR)
// CHECK-LABEL:   func.func @ptr_add_sgpr_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 31 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_ashrrev_i32_e64 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[CONSTANT_1]], %[[ARG1]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop3 v_lshl_add_u64 outs %[[MAKE_REGISTER_RANGE_2]] ins %[[ARG0]], %[[CONSTANT_0]] src2 = %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, i32, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_sgpr_dynamic(%ptr: !amdgcn.sgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr d_off = %off : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr<[? + 2]>
}

// SGPR ptr + uniform offset only
// CHECK-LABEL:   func.func @ptr_add_sgpr_uniform(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 31 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_ashr_i32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[ARG1]], %[[CONSTANT_0]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_2]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_3:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_add_u32 outs %[[SPLIT_REGISTER_RANGE_1]]#0 ins %[[SPLIT_REGISTER_RANGE_2]]#0, %[[SPLIT_REGISTER_RANGE_3]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_2:.*]] = amdgcn.sop2 s_addc_u32 outs %[[SPLIT_REGISTER_RANGE_1]]#1 ins %[[SPLIT_REGISTER_RANGE_2]]#1, %[[SPLIT_REGISTER_RANGE_3]]#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_3:.*]] = amdgcn.make_register_range %[[VAL_1]], %[[VAL_2]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           return %[[MAKE_REGISTER_RANGE_3]] : !amdgcn.sgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_sgpr_uniform(%ptr: !amdgcn.sgpr<[? + 2]>, %uoff: !amdgcn.sgpr) -> !amdgcn.sgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr u_off = %uoff : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr
  return %0 : !amdgcn.sgpr<[? + 2]>
}

// SGPR ptr + const + dynamic
// CHECK-LABEL:   func.func @ptr_add_sgpr_const_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 31 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 32 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_0]] ins %[[CONSTANT_2]] : !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_ashr_i32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[VAL_0]], %[[CONSTANT_1]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_2]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_3:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[VAL_2:.*]] = amdgcn.sop2 s_add_u32 outs %[[SPLIT_REGISTER_RANGE_1]]#0 ins %[[SPLIT_REGISTER_RANGE_2]]#0, %[[SPLIT_REGISTER_RANGE_3]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_3:.*]] = amdgcn.sop2 s_addc_u32 outs %[[SPLIT_REGISTER_RANGE_1]]#1 ins %[[SPLIT_REGISTER_RANGE_2]]#1, %[[SPLIT_REGISTER_RANGE_3]]#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_3:.*]] = amdgcn.make_register_range %[[VAL_2]], %[[VAL_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_4:.*]] = amdgcn.make_register_range %[[ALLOCA_5]], %[[ALLOCA_6]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_4:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_4]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[COPY_1:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_4]]#0, %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_4:.*]] = amdgcn.vop3 v_ashrrev_i32_e64 outs %[[SPLIT_REGISTER_RANGE_4]]#1 ins %[[CONSTANT_1]], %[[ARG1]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_5:.*]] = amdgcn.make_register_range %[[COPY_1]], %[[VAL_4]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_8:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_6:.*]] = amdgcn.make_register_range %[[ALLOCA_7]], %[[ALLOCA_8]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_5:.*]] = amdgcn.vop3 v_lshl_add_u64 outs %[[MAKE_REGISTER_RANGE_6]] ins %[[MAKE_REGISTER_RANGE_3]], %[[CONSTANT_0]] src2 = %[[MAKE_REGISTER_RANGE_5]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, i32, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_5]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_sgpr_const_dynamic(%ptr: !amdgcn.sgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr d_off = %off c_off = 32 : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr<[? + 2]>
}

// VGPR ptr (i32) + constant offset only
// CHECK-LABEL:   func.func @ptr_add_vgpr_i32_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 64 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_add_u32 outs %[[ALLOCA_0]] ins %[[CONSTANT_0]], %[[ARG0]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @ptr_add_vgpr_i32_const(%ptr: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %ptr c_off = 64 : !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

// VGPR ptr (i32) + dynamic offset only
// CHECK-LABEL:   func.func @ptr_add_vgpr_i32_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_add_u32 outs %[[ALLOCA_0]] ins %[[ARG0]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_0]] : !amdgcn.vgpr
// CHECK:         }
func.func @ptr_add_vgpr_i32_dynamic(%ptr: !amdgcn.vgpr, %off: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %ptr d_off = %off : !amdgcn.vgpr, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

// VGPR ptr (i64) + constant offset only
// CHECK-LABEL:   func.func @ptr_add_vgpr_i64_const(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 31 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 32 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop1 s_mov_b32 outs %[[ALLOCA_0]] ins %[[CONSTANT_2]] : !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop3 v_ashrrev_i32_e64 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[CONSTANT_1]], %[[VAL_0]] : !amdgcn.vgpr, i32, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]] = amdgcn.vop3 v_lshl_add_u64 outs %[[MAKE_REGISTER_RANGE_2]] ins %[[ARG0]], %[[CONSTANT_0]] src2 = %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, i32, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_2]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_vgpr_i64_const(%ptr: !amdgcn.vgpr<[? + 2]>) -> !amdgcn.vgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr c_off = 32 : !amdgcn.vgpr<[? + 2]>
  return %0 : !amdgcn.vgpr<[? + 2]>
}

// VGPR ptr (i64) + dynamic offset only
// CHECK-LABEL:   func.func @ptr_add_vgpr_i64_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 31 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop3 v_ashrrev_i32_e64 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[CONSTANT_1]], %[[ARG1]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop3 v_lshl_add_u64 outs %[[MAKE_REGISTER_RANGE_2]] ins %[[ARG0]], %[[CONSTANT_0]] src2 = %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, i32, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_1]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_vgpr_i64_dynamic(%ptr: !amdgcn.vgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr d_off = %off : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr<[? + 2]>
}

// VGPR ptr (i64) + const + dynamic
// CHECK-LABEL:   func.func @ptr_add_vgpr_i64_const_dynamic(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 31 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 8 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop2 v_add_u32 outs %[[ALLOCA_0]] ins %[[CONSTANT_2]], %[[ARG1]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop3 v_ashrrev_i32_e64 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[CONSTANT_1]], %[[VAL_0]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]] = amdgcn.vop3 v_lshl_add_u64 outs %[[MAKE_REGISTER_RANGE_2]] ins %[[ARG0]], %[[CONSTANT_0]] src2 = %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, i32, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_2]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_vgpr_i64_const_dynamic(%ptr: !amdgcn.vgpr<[? + 2]>, %off: !amdgcn.vgpr) -> !amdgcn.vgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr d_off = %off c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr
  return %0 : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @ptr_add_vgpr_i32_all(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr, %[[ARG1:.*]]: !amdgcn.vgpr, %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.vgpr {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_add_u32 outs %[[ALLOCA_0]] ins %[[ARG2]], %[[CONSTANT_0]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop2 v_add_u32 outs %[[ALLOCA_1]] ins %[[VAL_0]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]] = amdgcn.vop2 v_add_u32 outs %[[ALLOCA_2]] ins %[[ARG0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           return %[[VAL_2]] : !amdgcn.vgpr
// CHECK:         }
func.func @ptr_add_vgpr_i32_all(%ptr: !amdgcn.vgpr, %d_off: !amdgcn.vgpr, %u_off: !amdgcn.sgpr) -> !amdgcn.vgpr {
  %0 = amdgcn.ptr_add %ptr d_off = %d_off u_off = %u_off c_off = 8 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr
  return %0 : !amdgcn.vgpr
}

// CHECK-LABEL:   func.func @ptr_add_vgpr_i64_all(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.vgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 31 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 8 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_add_u32 outs %[[ALLOCA_0]] ins %[[ARG2]], %[[CONSTANT_2]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop2 v_add_u32 outs %[[ALLOCA_1]] ins %[[VAL_0]], %[[ARG1]] : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_2:.*]] = amdgcn.vop3 v_ashrrev_i32_e64 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[CONSTANT_1]], %[[VAL_1]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_4]], %[[ALLOCA_5]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_3:.*]] = amdgcn.vop3 v_lshl_add_u64 outs %[[MAKE_REGISTER_RANGE_2]] ins %[[ARG0]], %[[CONSTANT_0]] src2 = %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, i32, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_3]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_vgpr_i64_all(%ptr: !amdgcn.vgpr<[? + 2]>, %d_off: !amdgcn.vgpr, %u_off: !amdgcn.sgpr) -> !amdgcn.vgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr d_off = %d_off u_off = %u_off c_off = 8 : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, !amdgcn.sgpr
  return %0 : !amdgcn.vgpr<[? + 2]>
}

// CHECK-LABEL:   func.func @ptr_add_sgpr_i64_all(
// CHECK-SAME:      %[[ARG0:.*]]: !amdgcn.sgpr<[? + 2]>,
// CHECK-SAME:      %[[ARG1:.*]]: !amdgcn.vgpr,
// CHECK-SAME:      %[[ARG2:.*]]: !amdgcn.sgpr) -> !amdgcn.vgpr<[? + 2]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 31 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 8 : i32
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.sop2 s_add_u32 outs %[[ALLOCA_0]] ins %[[ARG2]], %[[CONSTANT_2]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_1]], %[[ALLOCA_2]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_0:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[COPY_0:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_0]]#0, %[[VAL_0]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.sop2 s_ashr_i32 outs %[[SPLIT_REGISTER_RANGE_0]]#1 ins %[[VAL_0]], %[[CONSTANT_1]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[COPY_0]], %[[VAL_1]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_2:.*]] = amdgcn.make_register_range %[[ALLOCA_3]], %[[ALLOCA_4]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_1:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_2]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_2:.*]]:2 = amdgcn.split_register_range %[[ARG0]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[SPLIT_REGISTER_RANGE_3:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.sgpr<[? + 2]>
// CHECK:           %[[VAL_2:.*]] = amdgcn.sop2 s_add_u32 outs %[[SPLIT_REGISTER_RANGE_1]]#0 ins %[[SPLIT_REGISTER_RANGE_2]]#0, %[[SPLIT_REGISTER_RANGE_3]]#0 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[VAL_3:.*]] = amdgcn.sop2 s_addc_u32 outs %[[SPLIT_REGISTER_RANGE_1]]#1 ins %[[SPLIT_REGISTER_RANGE_2]]#1, %[[SPLIT_REGISTER_RANGE_3]]#1 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_3:.*]] = amdgcn.make_register_range %[[VAL_2]], %[[VAL_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_4:.*]] = amdgcn.make_register_range %[[ALLOCA_5]], %[[ALLOCA_6]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[SPLIT_REGISTER_RANGE_4:.*]]:2 = amdgcn.split_register_range %[[MAKE_REGISTER_RANGE_4]] : !amdgcn.vgpr<[? + 2]>
// CHECK:           %[[COPY_1:.*]] = lsir.copy %[[SPLIT_REGISTER_RANGE_4]]#0, %[[ARG1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_4:.*]] = amdgcn.vop3 v_ashrrev_i32_e64 outs %[[SPLIT_REGISTER_RANGE_4]]#1 ins %[[CONSTANT_1]], %[[ARG1]] : !amdgcn.vgpr, i32, !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_5:.*]] = amdgcn.make_register_range %[[COPY_1]], %[[VAL_4]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_8:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_6:.*]] = amdgcn.make_register_range %[[ALLOCA_7]], %[[ALLOCA_8]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[VAL_5:.*]] = amdgcn.vop3 v_lshl_add_u64 outs %[[MAKE_REGISTER_RANGE_6]] ins %[[MAKE_REGISTER_RANGE_3]], %[[CONSTANT_0]] src2 = %[[MAKE_REGISTER_RANGE_5]] : !amdgcn.vgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>, i32, !amdgcn.vgpr<[? + 2]>
// CHECK:           return %[[VAL_5]] : !amdgcn.vgpr<[? + 2]>
// CHECK:         }
func.func @ptr_add_sgpr_i64_all(%ptr: !amdgcn.sgpr<[? + 2]>, %d_off: !amdgcn.vgpr, %u_off: !amdgcn.sgpr) -> !amdgcn.vgpr<[? + 2]> {
  %0 = amdgcn.ptr_add %ptr d_off = %d_off u_off = %u_off c_off = 8 : !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, !amdgcn.sgpr
  return %0 : !amdgcn.vgpr<[? + 2]>
}

// -----

// This test checks a regression where the arguments where getting invalid indices.
// CHECK-LABEL: kernel @test_load
// CHECK: load_arg 0 : !amdgcn.sgpr<[? + 2]>
module attributes {dlti.dl_spec = #dlti.dl_spec<!ptr.ptr<#amdgcn.addr_space<local, read_write>> = #ptr.spec<size = 32, abi = 32, preferred = 32>, !ptr.ptr<#amdgcn.addr_space<global, read_write>> = #ptr.spec<size = 64, abi = 64, preferred = 64>>} {
  amdgcn.module @test_module target = <gfx942> isa = <cdna3> {
    func.func @test_load(%arg0: !amdgcn.sgpr<[? + 2]>) attributes {gpu.host_abi = {align = array<i32: 8>, size = array<i32: 8>, type = (!ptr.ptr<#amdgcn.addr_space<global, read_write>>) -> ()}, gpu.kernel} {
      %c10_i32 = arith.constant 10 : i32
      %c2_i32 = arith.constant 2 : i32
      %c64_i32 = arith.constant 64 : i32
      %0 = amdgcn.thread_id  x : !amdgcn.vgpr
      %1 = amdgcn.block_id  x : !amdgcn.sgpr
      %2 = lsir.alloca : !amdgcn.vgpr
      %3 = lsir.shli i32 %2, %0, %c2_i32 : !amdgcn.vgpr, !amdgcn.vgpr, i32
      %4 = lsir.alloca : !amdgcn.sgpr
      %5 = lsir.shli i32 %4, %1, %c10_i32 : !amdgcn.sgpr, !amdgcn.sgpr, i32
      %6 = lsir.alloca : !amdgcn.vgpr
      %7 = lsir.addi i32 %6, %3, %5 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr
      %8 = lsir.alloca : !amdgcn.vgpr
      %9 = lsir.addi i32 %8, %7, %c64_i32 : !amdgcn.vgpr, !amdgcn.vgpr, i32
      %10 = amdgcn.alloca : !amdgcn.vgpr
      %dest_res, %token = amdgcn.load global_load_dword dest %10 addr %arg0 offset d(%9) : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.read_token<flat>
      amdgcn.test_inst ins %dest_res : (!amdgcn.vgpr) -> ()
      return
    }
  }
}
