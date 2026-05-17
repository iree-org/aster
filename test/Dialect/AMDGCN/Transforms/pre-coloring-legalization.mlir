// RUN: aster-opt %s --amdgcn-pre-coloring-legalization --split-input-file | FileCheck %s

// Verify that lsir.copy from an SGPR to SCC is replaced with s_cmp_eq_u32.
// CHECK-LABEL: func.func @sgpr_to_scc
// CHECK:         %[[SCC:.*]] = amdgcn.alloca : !amdgcn.scc<0>
// CHECK:         %[[SGPR:.*]] = amdgcn.alloca : !amdgcn.sgpr<?>
// CHECK:         %[[C1:.*]] = arith.constant 1 : i32
// CHECK:         amdgcn.s_cmp_eq_u32 outs(%[[SCC]]) ins(%[[SGPR]], %[[C1]])
// CHECK-NOT:     lsir.copy
func.func @sgpr_to_scc() {
  %scc = amdgcn.alloca : !amdgcn.scc<0>
  %sgpr = amdgcn.alloca : !amdgcn.sgpr<?>
  lsir.copy %scc, %sgpr : !amdgcn.scc<0>, !amdgcn.sgpr<?>
  return
}

// -----

// Verify that lsir.copy from VGPR to VGPR is not changed.
// CHECK-LABEL: func.func @vgpr_to_vgpr
// CHECK:         lsir.copy
func.func @vgpr_to_vgpr() {
  %dst = amdgcn.alloca : !amdgcn.vgpr<?>
  %src = amdgcn.alloca : !amdgcn.vgpr<?>
  lsir.copy %dst, %src : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  amdgcn.test_inst ins %dst : (!amdgcn.vgpr<?>) -> ()
  return
}

// -----

// Verify that lsir.copy from SCC to SGPR (the opposite direction) is not
// changed.
// CHECK-LABEL: func.func @scc_to_sgpr
// CHECK:         lsir.copy
func.func @scc_to_sgpr() {
  %scc = amdgcn.alloca : !amdgcn.scc<0>
  %sgpr = amdgcn.alloca : !amdgcn.sgpr<?>
  amdgcn.s_cmp_eq_i32 outs(%scc) ins(%sgpr, %sgpr) : outs(!amdgcn.scc<0>) ins(!amdgcn.sgpr<?>, !amdgcn.sgpr<?>)
  %promo = amdgcn.alloca : !amdgcn.sgpr<?>
  lsir.copy %promo, %scc : !amdgcn.sgpr<?>, !amdgcn.scc<0>
  amdgcn.test_inst ins %promo : (!amdgcn.sgpr<?>) -> ()
  return
}
