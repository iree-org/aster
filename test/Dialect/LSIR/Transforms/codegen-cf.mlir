// RUN: aster-opt %s --aster-codegen | FileCheck %s

//===----------------------------------------------------------------------===//
// Test CF dialect patterns: arith.cmpi/cmpf conversion and block argument
// handling in control flow operations. Verifies that:
// 1. arith.cmpi/cmpf are converted to lsir.cmpi/cmpf returning i1
// 2. Scalar block arguments are converted to register types
// 3. Branch operands are properly wrapped in alloca+mov when needed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: amdgcn.module @test_uniform_loop
// CHECK:         kernel @test_uniform_loop
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[LOAD:.*]] = load_arg 1
// CHECK:           amdgcn.sopp.s_waitcnt
// CHECK:           split_register_range
// CHECK:           assume_uniform
// CHECK:           %[[CMP_INIT:.*]] = lsir.cmpi i32 sgt %{{.*}}, %[[C0]] : !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_INIT:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[MOV_INIT:.*]] = lsir.mov %[[ALLOCA_INIT]], %[[C0]]
// CHECK:           cf.cond_br %[[CMP_INIT]], ^bb1(%[[MOV_INIT]] : !amdgcn.sgpr), ^bb2
// CHECK:         ^bb1(%[[LOOP_ARG:.*]]: !amdgcn.sgpr):
// CHECK:           test_inst ins %[[LOOP_ARG]]
// CHECK:           %[[ALLOCA_LOOP:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[LOOP_ADDI:.*]] = lsir.addi i32 %[[ALLOCA_LOOP]], %[[LOOP_ARG]], %[[C1]]
// CHECK:           %[[CMP_LOOP:.*]] = lsir.cmpi i32 slt %[[LOOP_ADDI]], %{{.*}} : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           cf.cond_br %[[CMP_LOOP]], ^bb1(%[[LOOP_ADDI]] : !amdgcn.sgpr), ^bb2
// CHECK:         ^bb2:
// CHECK:           end_kernel

amdgcn.module @test_uniform_loop target = <gfx942> isa = <cdna3> {
  kernel @test_uniform_loop arguments <[#amdgcn.buffer_arg<address_space = generic, access = read_only>, #amdgcn.buffer_arg<address_space = generic>]> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = amdgcn.load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %arg0, %arg1 = amdgcn.split_register_range %0 : !amdgcn.sgpr_range<[? + 2]>
    %1 = aster_utils.assume_uniform %arg0 : !amdgcn.sgpr
    %2 = lsir.from_reg %1 : !amdgcn.sgpr -> i32
    %3 = arith.cmpi sgt, %2, %c0_i32 : i32
    cf.cond_br %3, ^bb1(%c0_i32 : i32), ^bb2
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb1
    %5 = lsir.to_reg %4 : i32 -> !amdgcn.sgpr
    amdgcn.test_inst ins %5 : (!amdgcn.sgpr) -> ()
    %6 = arith.addi %4, %c1_i32 : i32
    %7 = arith.cmpi slt, %6, %2 : i32
    cf.cond_br %7, ^bb1(%6 : i32), ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    end_kernel
  }
}

// -----

// CHECK-LABEL: amdgcn.module @test_uniform_loop_with_load
// CHECK:         kernel @test_uniform_loop_with_load
// CHECK:           %[[C2:.*]] = arith.constant 2 : i32
// CHECK:           %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           load_arg 0
// CHECK:           load_arg 1
// CHECK:           amdgcn.sopp.s_waitcnt
// CHECK:           alloca
// CHECK:           %[[LOAD_RESULT:.*]], %{{.*}} = load s_load_dword
// CHECK:           amdgcn.sopp.s_waitcnt
// CHECK:           %[[CMP_INIT2:.*]] = lsir.cmpi i32 sgt %[[LOAD_RESULT]], %[[C0]] : !amdgcn.sgpr, i32
// CHECK:           %[[ALLOCA_INIT2:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[MOV_INIT2:.*]] = lsir.mov %[[ALLOCA_INIT2]], %[[C0]]
// CHECK:           cf.cond_br %[[CMP_INIT2]], ^bb1(%[[MOV_INIT2]] : !amdgcn.sgpr), ^bb2
// CHECK:         ^bb1(%[[LOOP_ARG2:.*]]: !amdgcn.sgpr):
// CHECK:           %[[ALLOCA_SHLI:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[LOOP_SHLI:.*]] = lsir.shli i32 %[[ALLOCA_SHLI]], %[[LOOP_ARG2]], %[[C2]]
// CHECK:           alloca
// CHECK:           amdgcn.vop1.vop1 <v_mov_b32_e32>
// CHECK:           store global_store_dword
// CHECK:           %[[ALLOCA_ADDI:.*]] = lsir.alloca : !amdgcn.sgpr
// CHECK:           %[[LOOP_ADDI2:.*]] = lsir.addi i32 %[[ALLOCA_ADDI]], %[[LOOP_ARG2]], %[[C1]]
// CHECK:           %[[CMP_LOOP2:.*]] = lsir.cmpi i32 slt %[[LOOP_ADDI2]], %[[LOAD_RESULT]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:           cf.cond_br %[[CMP_LOOP2]], ^bb1(%[[LOOP_ADDI2]] : !amdgcn.sgpr), ^bb2
// CHECK:         ^bb2:
// CHECK:           end_kernel

amdgcn.module @test_uniform_loop_with_load target = <gfx942> isa = <cdna3> {
  kernel @test_uniform_loop_with_load arguments <[#amdgcn.buffer_arg<address_space = generic, access = read_only>, #amdgcn.buffer_arg<address_space = generic>]> {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %1 = amdgcn.load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %2 = amdgcn.alloca : !amdgcn.sgpr
    %result, %token = amdgcn.load s_load_dword dest %2 addr %0 : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[? + 2]>) -> !amdgcn.read_token<constant>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    %3 = lsir.from_reg %result : !amdgcn.sgpr -> i32
    %4 = arith.cmpi sgt, %3, %c0_i32 : i32
    cf.cond_br %4, ^bb1(%c0_i32 : i32), ^bb2
  ^bb1(%5: i32):  // 2 preds: ^bb0, ^bb1
    %6 = arith.shli %5, %c2_i32 : i32
    %7 = lsir.to_reg %6 : i32 -> !amdgcn.sgpr
    %8 = amdgcn.alloca : !amdgcn.vgpr
    %9 = amdgcn.vop1.vop1 <v_mov_b32_e32> %8, %7 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %10 = amdgcn.store global_store_dword data %9 addr %1 offset d(%9) : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.vgpr) -> !amdgcn.write_token<flat>
    %11 = arith.addi %5, %c1_i32 : i32
    %12 = arith.cmpi slt, %11, %3 : i32
    cf.cond_br %12, ^bb1(%11 : i32), ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    end_kernel
  }
}
