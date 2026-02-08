// RUN: aster-opt %s --amdgcn-bufferization | FileCheck %s

// CHECK-LABEL:   amdgcn.module @too_few_allocas target = <gfx942> isa = <cdna3> {
// CHECK:           kernel @too_few_allocas {
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_0:.*]] = lsir.copy %[[VAL_1]], %[[VAL_0]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_2:.*]] = test_inst outs %[[VAL_0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_3:.*]] = test_inst outs %[[COPY_0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             %[[VAL_4:.*]] = alloca : !amdgcn.vgpr
// CHECK:             %[[COPY_1:.*]] = lsir.copy %[[VAL_4]], %[[VAL_2]] : !amdgcn.vgpr, !amdgcn.vgpr
// CHECK:             %[[VAL_5:.*]] = test_inst outs %[[VAL_2]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
// CHECK:             test_inst ins %[[COPY_1]], %[[VAL_3]], %[[VAL_5]] : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
// CHECK:             end_kernel
// CHECK:           }
// CHECK:         }
amdgcn.module @too_few_allocas target = <gfx942> isa = <cdna3> {
  kernel @too_few_allocas {
    %0 = alloca : !amdgcn.vgpr
    %1 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %2 = test_inst outs %0 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    %3 = test_inst outs %1 : (!amdgcn.vgpr) -> !amdgcn.vgpr
    test_inst ins %1, %2, %3 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }
}
