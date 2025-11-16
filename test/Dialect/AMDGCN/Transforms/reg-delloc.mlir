// RUN: aster-opt %s --amdgcn-register-dealloc | FileCheck %s

// CHECK-LABEL:   func.func @allocated_csed_mov() -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_1]], %[[ALLOCA_0]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_1]], %[[ALLOCA_0]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }

func.func @allocated_csed_mov() -> (!amdgcn.vgpr<2>, !amdgcn.vgpr<2>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<1>
  %1 = amdgcn.alloca : !amdgcn.vgpr<2>
  %2 = amdgcn.vop1.vop1 <v_mov_b32_e32> %1, %0 : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
  %3 = amdgcn.vop1.vop1 <v_mov_b32_e32> %1, %0 : (!amdgcn.vgpr<2>, !amdgcn.vgpr<1>) -> !amdgcn.vgpr<2>
  return %2, %3 : !amdgcn.vgpr<2>, !amdgcn.vgpr<2>
}

// CHECK-LABEL:   amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
amdgcn.module @ds_kernels target = <gfx942> isa = <cdna3> {
// CHECK-LABEL:     kernel @ds_all_kernel {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_2:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_3:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_5:.*]] = make_register_range %[[VAL_1]] : !amdgcn.vgpr
// CHECK:             %[[VAL_6:.*]] = amdgcn.ds.read <ds_read_b32> %[[VAL_5]], %[[VAL_0]] : !amdgcn.vgpr -> <[? + 1]>
// CHECK:             amdgcn.ds.write <ds_write_b32> %[[VAL_5]], %[[VAL_0]] : <[? + 1]>, !amdgcn.vgpr
// CHECK:             %[[VAL_7:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_8:.*]] = amdgcn.ds.read <ds_read_b64> %[[VAL_7]], %[[VAL_0]] : !amdgcn.vgpr -> <[? + 2]>
// CHECK:             amdgcn.ds.write <ds_write_b64> %[[VAL_7]], %[[VAL_0]] : <[? + 2]>, !amdgcn.vgpr
// CHECK:             %[[VAL_9:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_10:.*]] = amdgcn.ds.read <ds_read_b96> %[[VAL_9]], %[[VAL_0]], offset = 4 : !amdgcn.vgpr -> <[? + 3]>
// CHECK:             amdgcn.ds.write <ds_write_b96> %[[VAL_9]], %[[VAL_0]], offset = 4 : <[? + 3]>, !amdgcn.vgpr
// CHECK:             %[[VAL_11:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_12:.*]] = amdgcn.ds.read <ds_read_b128> %[[VAL_11]], %[[VAL_0]], offset = 8 : !amdgcn.vgpr -> <[? + 4]>
// CHECK:             amdgcn.ds.write <ds_write_b128> %[[VAL_11]], %[[VAL_0]], offset = 8 : <[? + 4]>, !amdgcn.vgpr
// CHECK:             end_kernel
// CHECK:           }
  kernel @ds_all_kernel {
    %0 = alloca : !amdgcn.vgpr<10>
    %1 = alloca : !amdgcn.vgpr<12>
    %2 = alloca : !amdgcn.vgpr<13>
    %3 = alloca : !amdgcn.vgpr<14>
    %4 = alloca : !amdgcn.vgpr<15>
    %5 = make_register_range %1 : !amdgcn.vgpr<12>
    %6 = amdgcn.ds.read <ds_read_b32> %5, %0 : <10> -> <[12 : 13]>
    amdgcn.ds.write <ds_write_b32> %5, %0 : <[12 : 13]>, <10>
    %7 = make_register_range %1, %2 : !amdgcn.vgpr<12>, !amdgcn.vgpr<13>
    %8 = amdgcn.ds.read <ds_read_b64> %7, %0 : <10> -> <[12 : 14]>
    amdgcn.ds.write <ds_write_b64> %7, %0 : <[12 : 14]>, <10>
    %9 = make_register_range %1, %2, %3 : !amdgcn.vgpr<12>, !amdgcn.vgpr<13>, !amdgcn.vgpr<14>
    %10 = amdgcn.ds.read <ds_read_b96> %9, %0, offset = 4 : <10> -> <[12 : 15]>
    amdgcn.ds.write <ds_write_b96> %9, %0, offset = 4 : <[12 : 15]>, <10>
    %11 = make_register_range %1, %2, %3, %4 : !amdgcn.vgpr<12>, !amdgcn.vgpr<13>, !amdgcn.vgpr<14>, !amdgcn.vgpr<15>
    %12 = amdgcn.ds.read <ds_read_b128> %11, %0, offset = 8 : <10> -> <[12 : 16]>
    amdgcn.ds.write <ds_write_b128> %11, %0, offset = 8 : <[12 : 16]>, <10>
    end_kernel
  }
}

//===----------------------------------------------------------------------===//
// Test CallOp conversion across function boundaries
//===----------------------------------------------------------------------===//

// CHECK-LABEL:   func.func @callee(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_0]], %arg0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_1]], %arg1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }

func.func @callee(%arg0: !amdgcn.vgpr<4>, %arg1: !amdgcn.vgpr<5>) -> (!amdgcn.vgpr<6>, !amdgcn.vgpr<7>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<6>
  %1 = amdgcn.alloca : !amdgcn.vgpr<7>
  %2 = amdgcn.vop1.vop1 <v_mov_b32_e32> %0, %arg0 : (!amdgcn.vgpr<6>, !amdgcn.vgpr<4>) -> !amdgcn.vgpr<6>
  %3 = amdgcn.vop1.vop1 <v_mov_b32_e32> %1, %arg1 : (!amdgcn.vgpr<7>, !amdgcn.vgpr<5>) -> !amdgcn.vgpr<7>
  return %2, %3 : !amdgcn.vgpr<6>, !amdgcn.vgpr<7>
}

// CHECK-LABEL:   func.func @test_call_op(%arg0: !amdgcn.vgpr, %arg1: !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[CALL_RESULT:.*]]:2 = call @callee(%arg0, %arg1) : (!amdgcn.vgpr, !amdgcn.vgpr) -> (!amdgcn.vgpr, !amdgcn.vgpr)
// CHECK:           %[[VAL_0:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_0]], %[[CALL_RESULT]]#0 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           %[[VAL_1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[ALLOCA_1]], %[[CALL_RESULT]]#1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:         }

func.func @test_call_op(%arg0: !amdgcn.vgpr<4>, %arg1: !amdgcn.vgpr<5>) -> (!amdgcn.vgpr<10>, !amdgcn.vgpr<11>) {
  %0 = amdgcn.alloca : !amdgcn.vgpr<10>
  %1 = amdgcn.alloca : !amdgcn.vgpr<11>
  %2:2 = call @callee(%arg0, %arg1) : (!amdgcn.vgpr<4>, !amdgcn.vgpr<5>) -> (!amdgcn.vgpr<6>, !amdgcn.vgpr<7>)
  %3 = amdgcn.vop1.vop1 <v_mov_b32_e32> %0, %2#0 : (!amdgcn.vgpr<10>, !amdgcn.vgpr<6>) -> !amdgcn.vgpr<10>
  %4 = amdgcn.vop1.vop1 <v_mov_b32_e32> %1, %2#1 : (!amdgcn.vgpr<11>, !amdgcn.vgpr<7>) -> !amdgcn.vgpr<11>
  return %3, %4 : !amdgcn.vgpr<10>, !amdgcn.vgpr<11>
}
