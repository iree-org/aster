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
// CHECK:             %[[VAL_6:.*]], %{{.*}} = load ds_read_b32 dest %[[VAL_5]] addr %[[VAL_0]] : dps(!amdgcn.vgpr_range<[? + 1]>) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:             %{{.*}} = store ds_write_b32 data %[[VAL_5]] addr %[[VAL_0]] : ins(!amdgcn.vgpr_range<[? + 1]>, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:             %[[VAL_7:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_8:.*]], %{{.*}} = load ds_read_b64 dest %[[VAL_7]] addr %[[VAL_0]] : dps(!amdgcn.vgpr_range<[? + 2]>) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>
// CHECK:             %{{.*}} = store ds_write_b64 data %[[VAL_7]] addr %[[VAL_0]] : ins(!amdgcn.vgpr_range<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.write_token<shared>
// CHECK:             %[[VAL_9:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_10:.*]], %{{.*}} = load ds_read_b96 dest %[[VAL_9]] addr %[[VAL_0]] offset c(%{{.*}}) : dps(!amdgcn.vgpr_range<[? + 3]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:             %{{.*}} = store ds_write_b96 data %[[VAL_9]] addr %[[VAL_0]] offset c(%{{.*}}) : ins(!amdgcn.vgpr_range<[? + 3]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
// CHECK:             %[[VAL_11:.*]] = make_register_range %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_12:.*]], %{{.*}} = load ds_read_b128 dest %[[VAL_11]] addr %[[VAL_0]] offset c(%{{.*}}) : dps(!amdgcn.vgpr_range<[? + 4]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
// CHECK:             %{{.*}} = store ds_write_b128 data %[[VAL_11]] addr %[[VAL_0]] offset c(%{{.*}}) : ins(!amdgcn.vgpr_range<[? + 4]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
// CHECK:             end_kernel
// CHECK:           }
  kernel @ds_all_kernel {
    %0 = alloca : !amdgcn.vgpr<10>
    %1 = alloca : !amdgcn.vgpr<12>
    %2 = alloca : !amdgcn.vgpr<13>
    %3 = alloca : !amdgcn.vgpr<14>
    %4 = alloca : !amdgcn.vgpr<15>
    %5 = make_register_range %1 : !amdgcn.vgpr<12>
    %c0 = arith.constant 0 : i32
    %6, %tok6 = amdgcn.load ds_read_b32 dest %5 addr %0 : dps(!amdgcn.vgpr_range<[12 : 13]>) ins(!amdgcn.vgpr<10>) -> !amdgcn.read_token<shared>
    %tok6a = amdgcn.store ds_write_b32 data %5 addr %0 : ins(!amdgcn.vgpr_range<[12 : 13]>, !amdgcn.vgpr<10>) -> !amdgcn.write_token<shared>
    %7 = make_register_range %1, %2 : !amdgcn.vgpr<12>, !amdgcn.vgpr<13>
    %8, %tok8 = amdgcn.load ds_read_b64 dest %7 addr %0 : dps(!amdgcn.vgpr_range<[12 : 14]>) ins(!amdgcn.vgpr<10>) -> !amdgcn.read_token<shared>
    %tok8a = amdgcn.store ds_write_b64 data %7 addr %0 : ins(!amdgcn.vgpr_range<[12 : 14]>, !amdgcn.vgpr<10>) -> !amdgcn.write_token<shared>
    %c4_reg = arith.constant 4 : i32
    %c8_reg = arith.constant 8 : i32
    %9 = make_register_range %1, %2, %3 : !amdgcn.vgpr<12>, !amdgcn.vgpr<13>, !amdgcn.vgpr<14>
    %10, %tok10 = amdgcn.load ds_read_b96 dest %9 addr %0 offset c(%c4_reg) : dps(!amdgcn.vgpr_range<[12 : 15]>) ins(!amdgcn.vgpr<10>, i32) -> !amdgcn.read_token<shared>
    %tok10a = amdgcn.store ds_write_b96 data %9 addr %0 offset c(%c4_reg) : ins(!amdgcn.vgpr_range<[12 : 15]>, !amdgcn.vgpr<10>, i32) -> !amdgcn.write_token<shared>
    %11 = make_register_range %1, %2, %3, %4 : !amdgcn.vgpr<12>, !amdgcn.vgpr<13>, !amdgcn.vgpr<14>, !amdgcn.vgpr<15>
    %12, %tok12 = amdgcn.load ds_read_b128 dest %11 addr %0 offset c(%c8_reg) : dps(!amdgcn.vgpr_range<[12 : 16]>) ins(!amdgcn.vgpr<10>, i32) -> !amdgcn.read_token<shared>
    %tok12a = amdgcn.store ds_write_b128 data %11 addr %0 offset c(%c8_reg) : ins(!amdgcn.vgpr_range<[12 : 16]>, !amdgcn.vgpr<10>, i32) -> !amdgcn.write_token<shared>
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
