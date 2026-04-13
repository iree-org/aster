// RUN: aster-opt %s --split-input-file \
// RUN:   --amdgcn-lower-sreg-block-args \
// RUN:   | FileCheck %s

// CHECK-LABEL: kernel @scc_through_sgpr
// CHECK:         %[[SG:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[CPY0:.*]] = lsir.copy %[[SG]], %{{.*}} : !amdgcn.sgpr, !amdgcn.scc
// CHECK:         cf.br ^[[BB1:bb[0-9]+]](%[[CPY0]] : !amdgcn.sgpr)
// CHECK:       ^[[BB1]](%[[A:.*]]: !amdgcn.sgpr):
// CHECK:         %[[SC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[CPY1:.*]] = lsir.copy %[[SC]], %[[A]] : !amdgcn.scc{{.*}}, !amdgcn.sgpr
// CHECK:         test_inst ins %[[CPY1]] : (!amdgcn.scc) -> ()
// CHECK:         end_kernel
amdgcn.kernel @scc_through_sgpr {
^bb0:
  %scc = amdgcn.alloca : !amdgcn.scc
  cf.br ^bb1(%scc : !amdgcn.scc)
^bb1(%arg : !amdgcn.scc):
  test_inst ins %arg : (!amdgcn.scc) -> ()
  amdgcn.end_kernel
}

// -----

// CHECK-LABEL: kernel @cond_br_scc
// CHECK:         alloca : !amdgcn.sgpr
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.sgpr, !amdgcn.scc
// CHECK:         alloca : !amdgcn.sgpr
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.sgpr, !amdgcn.scc
// CHECK:         cf.cond_br %{{.*}}, ^{{bb[0-9]+}}(%{{.*}} : !amdgcn.sgpr), ^{{bb[0-9]+}}(%{{.*}} : !amdgcn.sgpr)
// CHECK:       ^bb{{[0-9]+}}(%{{.*}}: !amdgcn.sgpr):
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.scc{{.*}}, !amdgcn.sgpr
// CHECK:       ^bb{{[0-9]+}}(%{{.*}}: !amdgcn.sgpr):
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.scc{{.*}}, !amdgcn.sgpr
amdgcn.kernel @cond_br_scc {
^bb0:
  %c = arith.constant true
  %scc = amdgcn.alloca : !amdgcn.scc
  cf.cond_br %c, ^bb1(%scc : !amdgcn.scc), ^bb2(%scc : !amdgcn.scc)
^bb1(%a1 : !amdgcn.scc):
  amdgcn.end_kernel
^bb2(%a2 : !amdgcn.scc):
  amdgcn.end_kernel
}

// -----

// Allocated SCC (`<0>`) has non-value semantics; the pass must not rewrite.
// CHECK-LABEL: kernel @allocated_scc_unchanged
// CHECK:         cf.br ^{{bb[0-9]+}}(%{{.*}} : !amdgcn.scc<0>)
// CHECK:       ^{{bb[0-9]+}}(%{{.*}}: !amdgcn.scc<0>):
amdgcn.kernel @allocated_scc_unchanged {
^bb0:
  %scc = amdgcn.alloca : !amdgcn.scc<0>
  cf.br ^bb1(%scc : !amdgcn.scc<0>)
^bb1(%arg : !amdgcn.scc<0>):
  amdgcn.end_kernel
}

// -----

// Unallocated SCC (`<?>`) has non-value semantics; the pass must not rewrite.
// CHECK-LABEL: kernel @unallocated_scc_unchanged
// CHECK:         cf.br ^{{bb[0-9]+}}(%{{.*}} : !amdgcn.scc<?>)
// CHECK:       ^{{bb[0-9]+}}(%{{.*}}: !amdgcn.scc<?>):
amdgcn.kernel @unallocated_scc_unchanged {
^bb0:
  %scc = amdgcn.alloca : !amdgcn.scc<?>
  cf.br ^bb1(%scc : !amdgcn.scc<?>)
^bb1(%arg : !amdgcn.scc<?>):
  amdgcn.end_kernel
}

// -----

// VCC is a 64-bit (2-word) special register; the SGPR carrier must have size 2.
// CHECK-LABEL: kernel @vcc_through_sgpr
// CHECK:         %[[CPY0:.*]] = lsir.copy %{{.*}}, %{{.*}} : !amdgcn.sgpr{{.*}}, !amdgcn.vcc
// CHECK:         cf.br ^[[BB1:bb[0-9]+]](%[[CPY0]] : !amdgcn.sgpr<[? + 2]>)
// CHECK:       ^[[BB1]](%[[A:.*]]: !amdgcn.sgpr<[? + 2]>):
// CHECK:         %[[VC:.*]] = alloca : !amdgcn.vcc
// CHECK:         %[[CPY1:.*]] = lsir.copy %[[VC]], %[[A]] : !amdgcn.vcc{{.*}}, !amdgcn.sgpr{{.*}}
// CHECK:         end_kernel
amdgcn.kernel @vcc_through_sgpr {
^bb0:
  %vcc = amdgcn.alloca : !amdgcn.vcc
  cf.br ^bb1(%vcc : !amdgcn.vcc)
^bb1(%arg : !amdgcn.vcc):
  amdgcn.end_kernel
}

// -----

// Two distinct predecessors forwarding SCC to the same block argument.
// CHECK-LABEL: kernel @multi_pred_scc
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.sgpr, !amdgcn.scc
// CHECK:         cf.cond_br %{{.*}}, ^{{bb[0-9]+}}, ^[[DEST:bb[0-9]+]](%{{.*}} : !amdgcn.sgpr)
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.sgpr, !amdgcn.scc
// CHECK:         cf.br ^[[DEST]](%{{.*}} : !amdgcn.sgpr)
// CHECK:       ^[[DEST]](%{{.*}}: !amdgcn.sgpr):
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.scc{{.*}}, !amdgcn.sgpr
amdgcn.kernel @multi_pred_scc {
^bb0:
  %c = arith.constant true
  %s0 = amdgcn.alloca : !amdgcn.scc
  cf.cond_br %c, ^bb1, ^bb2(%s0 : !amdgcn.scc)
^bb1:
  %s1 = amdgcn.alloca : !amdgcn.scc
  cf.br ^bb2(%s1 : !amdgcn.scc)
^bb2(%arg : !amdgcn.scc):
  amdgcn.end_kernel
}
