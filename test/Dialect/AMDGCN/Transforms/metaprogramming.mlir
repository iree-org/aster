// RUN: aster-opt %s --pass-pipeline="builtin.module(func.func(aster-constexpr-expansion,cse,canonicalize,amdgcn-mem2reg))" | FileCheck %s


// This test checks perfect interleaving of registers when unrolled.
// Concretely, we check that the output of an instruction is used as
// input to the next instruction, alternating between two register ranges.
func.func @test_interleaving() -> !amdgcn.vgpr<[? + 4]> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %alloca = memref.alloca() : memref<!amdgcn.vgpr<[? + 4]>>
  %alloca_0 = memref.alloca() : memref<!amdgcn.vgpr<[? + 4]>>
  scf.for %arg0 = %c0 to %c5 step %c1 {
    %1 = arith.remsi %arg0, %c2 : index
    %2 = arith.cmpi eq, %1, %c0 : index
    %3:2 = scf.if %2 -> (memref<!amdgcn.vgpr<[? + 4]>>, memref<!amdgcn.vgpr<[? + 4]>>) {
      scf.yield %alloca, %alloca_0 : memref<!amdgcn.vgpr<[? + 4]>>, memref<!amdgcn.vgpr<[? + 4]>>
    } else {
      scf.yield %alloca_0, %alloca : memref<!amdgcn.vgpr<[? + 4]>>, memref<!amdgcn.vgpr<[? + 4]>>
    }
    %4 = memref.load %3#0[] : memref<!amdgcn.vgpr<[? + 4]>>
    %5 = memref.load %3#1[] : memref<!amdgcn.vgpr<[? + 4]>>
    %6 = amdgcn.test_inst outs %5 ins %4 : (!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>) -> !amdgcn.vgpr<[? + 4]>
    memref.store %6, %3#1[] : memref<!amdgcn.vgpr<[? + 4]>>
  } {aster.constexpr}
  %0 = memref.load %alloca[] : memref<!amdgcn.vgpr<[? + 4]>>
  return %0 : !amdgcn.vgpr<[? + 4]>
}
// CHECK-LABEL:   func.func @test_interleaving() -> !amdgcn.vgpr<[? + 4]> {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.vgpr
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[ALLOCA_4]], %[[ALLOCA_5]], %[[ALLOCA_6]], %[[ALLOCA_7]] : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[MAKE_REGISTER_RANGE_0]] ins %[[MAKE_REGISTER_RANGE_1]] : (!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>) -> !amdgcn.vgpr<[? + 4]>
// CHECK:           %[[TEST_INST_1:.*]] = amdgcn.test_inst outs %[[MAKE_REGISTER_RANGE_1]] ins %[[TEST_INST_0]] : (!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>) -> !amdgcn.vgpr<[? + 4]>
// CHECK:           %[[TEST_INST_2:.*]] = amdgcn.test_inst outs %[[TEST_INST_0]] ins %[[TEST_INST_1]] : (!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>) -> !amdgcn.vgpr<[? + 4]>
// CHECK:           %[[TEST_INST_3:.*]] = amdgcn.test_inst outs %[[TEST_INST_1]] ins %[[TEST_INST_2]] : (!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>) -> !amdgcn.vgpr<[? + 4]>
// CHECK:           %[[TEST_INST_4:.*]] = amdgcn.test_inst outs %[[TEST_INST_2]] ins %[[TEST_INST_3]] : (!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>) -> !amdgcn.vgpr<[? + 4]>
// CHECK:           return %[[TEST_INST_3]] : !amdgcn.vgpr<[? + 4]>
// CHECK:         }
