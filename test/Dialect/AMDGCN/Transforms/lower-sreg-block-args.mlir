// RUN: aster-opt %s --split-input-file \
// RUN:   --aster-amdgcn-bufferization \
// RUN:   | FileCheck %s

// CHECK-LABEL: kernel @scc_through_sgpr
// CHECK:         %[[CARR:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:         %[[ARG:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[SCC:.*]] = alloca : !amdgcn.scc
// CHECK:         cf.br ^[[FWD:bb[0-9]+]]
// CHECK:       ^[[FWD]]:
// CHECK:         lsir.copy %[[CARR]], %[[SCC]] : !amdgcn.sgpr<?>, !amdgcn.scc
// CHECK:         cf.br ^[[BB1:bb[0-9]+]]
// CHECK:       ^[[BB1]]:
// CHECK:         %[[CPY:.*]] = lsir.copy %[[ARG]], %[[CARR]] : !amdgcn.scc, !amdgcn.sgpr<?>
// CHECK:         test_inst ins %[[CPY]] : (!amdgcn.scc) -> ()
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
// CHECK:         alloca : !amdgcn.sgpr<?>
// CHECK:         alloca : !amdgcn.scc
// CHECK:         alloca : !amdgcn.sgpr<?>
// CHECK:         alloca : !amdgcn.scc
// CHECK:         alloca : !amdgcn.scc
// CHECK:         cf.cond_br %{{.*}}, ^{{bb[0-9]+}}, ^{{bb[0-9]+}}
// CHECK:       ^{{bb[0-9]+}}:
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, !amdgcn.scc
// CHECK:         cf.br ^{{bb[0-9]+}}
// CHECK:       ^{{bb[0-9]+}}:
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, !amdgcn.scc
// CHECK:         cf.br ^{{bb[0-9]+}}
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
// CHECK:         make_register_range %{{.*}}, %{{.*}} : !amdgcn.sgpr<?>, !amdgcn.sgpr<?>
// CHECK:         make_register_range %{{.*}}, %{{.*}} : !amdgcn.vcc_lo, !amdgcn.vcc_hi
// CHECK:         make_register_range %{{.*}}, %{{.*}} : !amdgcn.vcc_lo, !amdgcn.vcc_hi
// CHECK:         lsir.copy %{{.*}}, %{{.*}} : !amdgcn.sgpr<[? : ? + 2]>, !amdgcn.vcc
amdgcn.kernel @vcc_through_sgpr {
^bb0:
  %vcc_lo = amdgcn.alloca : !amdgcn.vcc_lo
  %vcc_hi = amdgcn.alloca : !amdgcn.vcc_hi
  %vcc = amdgcn.make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
  cf.br ^bb1(%vcc : !amdgcn.vcc)
^bb1(%arg : !amdgcn.vcc):
  amdgcn.end_kernel
}

// -----

// Two distinct predecessors forwarding SCC to the same block argument.
// CHECK-LABEL: kernel @multi_pred_scc
// CHECK:         %[[CARR:.*]] = alloca : !amdgcn.sgpr<?>
// CHECK:         %[[ARG:.*]] = alloca : !amdgcn.scc
// CHECK:         alloca : !amdgcn.scc
// CHECK:         cf.cond_br %{{.*}}, ^{{bb[0-9]+}}, ^{{bb[0-9]+}}
// CHECK:       ^{{bb[0-9]+}}:
// CHECK:         lsir.copy %[[CARR]], %{{.*}} : !amdgcn.sgpr<?>, !amdgcn.scc
// CHECK:         cf.br ^[[DEST:bb[0-9]+]]
// CHECK:       ^{{bb[0-9]+}}:
// CHECK:         alloca : !amdgcn.scc
// CHECK:       ^{{bb[0-9]+}}:
// CHECK:         lsir.copy %[[CARR]], %{{.*}} : !amdgcn.sgpr<?>, !amdgcn.scc
// CHECK:         cf.br ^[[DEST]]
// CHECK:       ^[[DEST]]:
// CHECK:         end_kernel
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
