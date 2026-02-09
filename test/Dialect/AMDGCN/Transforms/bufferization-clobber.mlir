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

// -----

// Same alloca written twice in ^bb0; the first value (%v1) is used in a
// successor block. The clobber copy must replace that cross-block use.
//
// CHECK-LABEL: @cross_block_clobber
//       CHECK:   %[[A:.*]] = alloca : !amdgcn.vgpr
//       CHECK:   %[[V1:.*]] = test_inst outs %[[A]]
//       CHECK:   %[[COPY_A:.*]] = alloca : !amdgcn.vgpr
//       CHECK:   %[[COPY:.*]] = amdgcn.vop1.vop1 <v_mov_b32_e32> %[[COPY_A]], %[[V1]]
//       CHECK:   %[[V2:.*]] = test_inst outs %[[A]]
//       CHECK:   cf.cond_br
//       CHECK: ^bb1:
//       CHECK:   test_inst ins %[[COPY]]
//       CHECK: ^bb2:
//       CHECK:   test_inst ins %[[V2]]
amdgcn.module @cross_block_clobber target = <gfx942> isa = <cdna3> {
  func.func private @rand() -> i1
  kernel @cross_block_clobber {
    %cond = func.call @rand() : () -> i1
    %0 = alloca : !amdgcn.vgpr
    %1 = alloca : !amdgcn.sgpr
    %v1 = test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    %v2 = test_inst outs %0 ins %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    test_inst ins %v1 : (!amdgcn.vgpr) -> ()
    cf.br ^bb3
  ^bb2:
    test_inst ins %v2 : (!amdgcn.vgpr) -> ()
    cf.br ^bb3
  ^bb3:
    end_kernel
  }
}
