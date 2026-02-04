// RUN: aster-opt %s --amdgcn-bufferization --split-input-file | FileCheck %s

// CHECK-LABEL: @too_few_allocas
//       CHECK:   %[[a0:.*]] = alloca : !amdgcn.vgpr
//       CHECK:   %[[t0:.*]] = test_inst outs %[[a0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:   %[[a1:.*]] = alloca : !amdgcn.vgpr
//       CHECK:   %[[c1:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[a1]], %[[t0]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:   %[[t2:.*]] = test_inst outs %[[a0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:   %[[a2:.*]] = alloca : !amdgcn.vgpr
//       CHECK:   %[[c2:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[a2]], %[[t2]] : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:   %[[t3:.*]] = test_inst outs %[[t0]] : (!amdgcn.vgpr) -> !amdgcn.vgpr
//       CHECK:   test_inst ins %[[c1]], %[[c2]], %[[t3]] : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) -> ()
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
