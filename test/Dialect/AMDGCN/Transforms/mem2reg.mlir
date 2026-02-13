// RUN: aster-opt %s --amdgcn-mem2reg | FileCheck %s

// CHECK-LABEL:   func.func @test_basic() -> !amdgcn.sgpr {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr
// CHECK:           %[[TEST_INST_0:.*]] = amdgcn.test_inst outs %[[ALLOCA_0]] ins %[[ALLOCA_1]] : (!amdgcn.sgpr, !amdgcn.sgpr) -> !amdgcn.sgpr
// CHECK:           return %[[TEST_INST_0]] : !amdgcn.sgpr
// CHECK:         }
func.func @test_basic() -> !amdgcn.sgpr {
  %alloca = memref.alloca() : memref<!amdgcn.sgpr>
  %alloca_0 = memref.alloca() : memref<!amdgcn.sgpr>
  %0 = memref.load %alloca[] : memref<!amdgcn.sgpr>
  %1 = memref.load %alloca_0[] : memref<!amdgcn.sgpr>
  %2 = amdgcn.test_inst outs %1 ins %0 : (!amdgcn.sgpr, !amdgcn.sgpr) -> !amdgcn.sgpr
  memref.store %2, %alloca[] : memref<!amdgcn.sgpr>
  %3 = memref.load %alloca[] : memref<!amdgcn.sgpr>
  return %3 : !amdgcn.sgpr
}

// CHECK-LABEL:   func.func @test_allocated() -> !amdgcn.sgpr_range<[0 : 4]> {
// CHECK:           %[[ALLOCA_0:.*]] = amdgcn.alloca : !amdgcn.sgpr<8>
// CHECK:           %[[ALLOCA_1:.*]] = amdgcn.alloca : !amdgcn.sgpr<9>
// CHECK:           %[[ALLOCA_2:.*]] = amdgcn.alloca : !amdgcn.sgpr<10>
// CHECK:           %[[ALLOCA_3:.*]] = amdgcn.alloca : !amdgcn.sgpr<11>
// CHECK:           %[[MAKE_REGISTER_RANGE_0:.*]] = amdgcn.make_register_range %[[ALLOCA_0]], %[[ALLOCA_1]], %[[ALLOCA_2]], %[[ALLOCA_3]] : !amdgcn.sgpr<8>, !amdgcn.sgpr<9>, !amdgcn.sgpr<10>, !amdgcn.sgpr<11>
// CHECK:           %[[ALLOCA_4:.*]] = amdgcn.alloca : !amdgcn.sgpr<0>
// CHECK:           %[[ALLOCA_5:.*]] = amdgcn.alloca : !amdgcn.sgpr<1>
// CHECK:           %[[ALLOCA_6:.*]] = amdgcn.alloca : !amdgcn.sgpr<2>
// CHECK:           %[[ALLOCA_7:.*]] = amdgcn.alloca : !amdgcn.sgpr<3>
// CHECK:           %[[MAKE_REGISTER_RANGE_1:.*]] = amdgcn.make_register_range %[[ALLOCA_4]], %[[ALLOCA_5]], %[[ALLOCA_6]], %[[ALLOCA_7]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
// CHECK:           amdgcn.test_inst outs %[[MAKE_REGISTER_RANGE_1]] ins %[[MAKE_REGISTER_RANGE_0]] : (!amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr_range<[8 : 12]>) -> ()
// CHECK:           return %[[MAKE_REGISTER_RANGE_1]] : !amdgcn.sgpr_range<[0 : 4]>
// CHECK:         }
func.func @test_allocated() -> !amdgcn.sgpr_range<[0 : 4]> {
  %alloca = memref.alloca() : memref<!amdgcn.sgpr_range<[0 : 4]>>
  %alloca_0 = memref.alloca() : memref<!amdgcn.sgpr_range<[8 : 12]>>
  %0 = memref.load %alloca[] : memref<!amdgcn.sgpr_range<[0 : 4]>>
  %1 = memref.load %alloca_0[] : memref<!amdgcn.sgpr_range<[8 : 12]>>
  amdgcn.test_inst outs %0 ins %1 : (!amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr_range<[8 : 12]>) -> ()
  memref.store %0, %alloca[] : memref<!amdgcn.sgpr_range<[0 : 4]>>
  %3 = memref.load %alloca[] : memref<!amdgcn.sgpr_range<[0 : 4]>>
  return %3 : !amdgcn.sgpr_range<[0 : 4]>
}
