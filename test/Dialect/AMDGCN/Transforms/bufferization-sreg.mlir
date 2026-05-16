// RUN: aster-opt %s --aster-amdgcn-bufferization --split-input-file | FileCheck %s

// Basic SCC clobber: two s_cmp_eq_i32 in sequence, first SCC used after second.
// The first SCC value must be promoted to SGPR.

// CHECK-LABEL: amdgcn.kernel @basic_scc_clobber {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[A:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[B:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[C:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]]) ins(%[[A]], %[[B]])
// CHECK:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// CHECK:         %[[SCC1:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]]) ins(%[[A]], %[[C]])
// CHECK:         test_inst ins %[[PROMOTED]], %[[SCC1]]
// CHECK:         end_kernel
amdgcn.kernel @basic_scc_clobber {
  %scc = alloca : !amdgcn.scc
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc0, %scc1 : (!amdgcn.scc, !amdgcn.scc) -> ()
  end_kernel
}

// -----

// No promotion needed: only the last SCC definition is used.
// The first s_cmp_eq_i32 is dead and removed by CSE.

// CHECK-LABEL: amdgcn.kernel @no_promotion_needed {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK-NOT:     lsir.copy
// CHECK:         %[[SCC1:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         test_inst ins %[[SCC1]]
// CHECK:         end_kernel
amdgcn.kernel @no_promotion_needed {
  %scc = alloca : !amdgcn.scc
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc1 : (!amdgcn.scc) -> ()
  end_kernel
}

// -----

// SCC clobber by s_and_b32: the SCC from s_cmp_eq_i32 must be promoted.
// The use of scc0 is a general (non-special) operand of test_inst, so it is
// amended to the promoted SGPR value.

// CHECK-LABEL: amdgcn.kernel @scc_clobber_s_and {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[A:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[B:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[C:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[DST:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]]) ins(%[[A]], %[[B]])
// CHECK:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// CHECK:         %[[AND_RES:.*]], %{{.*}} = s_and_b32 outs(%[[DST]], %[[SCC_ALLOC]]) ins(%[[A]], %[[C]])
// CHECK:         test_inst ins %[[PROMOTED]], %[[AND_RES]]
// CHECK:         end_kernel
amdgcn.kernel @scc_clobber_s_and {
  %scc = alloca : !amdgcn.scc
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %dst = alloca : !amdgcn.sgpr
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %and, %scc1 = s_and_b32 outs(%dst, %scc) ins(%a, %c) : outs(!amdgcn.sgpr, !amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc0, %and : (!amdgcn.scc, !amdgcn.sgpr) -> ()
  end_kernel
}

// -----

// Chain of SCC clobbers: three definitions with all used afterwards.
// scc0 and scc1 must be promoted; scc2 stays in SCC.

// CHECK-LABEL: amdgcn.kernel @chain_of_scc_clobbers {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[A:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[B:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[C:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[D:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]]) ins(%[[A]], %[[B]])
// CHECK:         %[[PROMO0_ALLOC:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMO0:.*]] = lsir.copy %[[PROMO0_ALLOC]], %[[SCC0]]
// CHECK:         %[[SCC1:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]]) ins(%[[A]], %[[C]])
// CHECK:         %[[PROMO1_ALLOC:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMO1:.*]] = lsir.copy %[[PROMO1_ALLOC]], %[[SCC1]]
// CHECK:         %[[SCC2:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]]) ins(%[[A]], %[[D]])
// CHECK:         test_inst ins %[[PROMO0]], %[[PROMO1]], %[[SCC2]]
// CHECK:         end_kernel
amdgcn.kernel @chain_of_scc_clobbers {
  %scc = alloca : !amdgcn.scc
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %d = alloca : !amdgcn.sgpr
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %scc2 = s_cmp_eq_i32 outs(%scc) ins(%a, %d) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc0, %scc1, %scc2 : (!amdgcn.scc, !amdgcn.scc, !amdgcn.scc) -> ()
  end_kernel
}

// -----

// Efficient codegen: only scc0 is promoted because scc1 is the latest
// definition and can stay in SCC.

// CHECK-LABEL: amdgcn.kernel @efficient_last_def {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[A:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[B:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[C:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]]) ins(%[[A]], %[[B]])
// CHECK:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// CHECK:         %[[SCC1:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]]) ins(%[[A]], %[[C]])
// CHECK:         test_inst ins %[[PROMOTED]], %[[SCC1]]
// CHECK:         end_kernel
amdgcn.kernel @efficient_last_def {
  %scc = alloca : !amdgcn.scc
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc0, %scc1 : (!amdgcn.scc, !amdgcn.scc) -> ()
  end_kernel
}

// -----

// VCC clobber: two v_cmp_eq_i32 in sequence, first VCC used after second.
// VCC is a 64-bit composite register promoted to a 2-word SGPR.

// CHECK-LABEL: amdgcn.kernel @vcc_clobber {
// CHECK:         %[[VCC_LO:.*]] = alloca : !amdgcn.vcc_lo
// CHECK:         %[[VCC_HI:.*]] = alloca : !amdgcn.vcc_hi
// CHECK:         %[[VCC:.*]] = make_register_range %[[VCC_LO]], %[[VCC_HI]]
// CHECK:         %[[V1:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[V2:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[VCC0:.*]] = v_cmp_eq_i32 outs(%[[VCC]])
// CHECK:         %[[PROMO_LO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMO_HI:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMO:.*]] = make_register_range %[[PROMO_LO]], %[[PROMO_HI]]
// CHECK:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[VCC0]]
// CHECK:         %[[VCC1:.*]] = v_cmp_eq_i32 outs(%[[VCC]])
// CHECK:         test_inst ins %[[PROMOTED]], %[[VCC1]]
// CHECK:         end_kernel
amdgcn.kernel @vcc_clobber {
  %vcc_lo = alloca : !amdgcn.vcc_lo
  %vcc_hi = alloca : !amdgcn.vcc_hi
  %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
  %a = alloca : !amdgcn.sgpr
  %v1 = alloca : !amdgcn.vgpr
  %v2 = alloca : !amdgcn.vgpr
  %vcc0 = v_cmp_eq_i32 outs(%vcc) ins(%a, %v1) : outs(!amdgcn.vcc) ins(!amdgcn.sgpr, !amdgcn.vgpr)
  %vcc1 = v_cmp_eq_i32 outs(%vcc) ins(%a, %v2) : outs(!amdgcn.vcc) ins(!amdgcn.sgpr, !amdgcn.vgpr)
  test_inst ins %vcc0, %vcc1 : (!amdgcn.vcc, !amdgcn.vcc) -> ()
  end_kernel
}

// -----

// lsir.cond_br materialization: a promoted SCC is materialized back to the
// special register for the branch condition.

// CHECK-LABEL: amdgcn.kernel @cond_br_materialization {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// CHECK:         %[[MATERIALIZED:.*]] = lsir.copy %[[SCC_ALLOC]], %[[PROMOTED]]
// CHECK:         lsir.cond_br %[[MATERIALIZED]] : !amdgcn.scc
amdgcn.kernel @cond_br_materialization {
  %scc = alloca : !amdgcn.scc
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc1 : (!amdgcn.scc) -> ()
  lsir.cond_br %scc0 : !amdgcn.scc, ^bb1, ^bb2
^bb1:
  end_kernel
^bb2:
  end_kernel
}

// -----

// Cross-block SCC clobber: SCC defined in entry, clobbered only in one
// successor. The other successor uses SCC directly without promotion.

func.func private @rand() -> i1
// CHECK-LABEL: amdgcn.kernel @cross_block_scc {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         cf.cond_br
// CHECK:       ^bb1:
// CHECK:         %[[PROMO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED:.*]] = lsir.copy %[[PROMO]], %[[SCC0]]
// CHECK:         %[[SCC1:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         test_inst ins %[[PROMOTED]], %[[SCC1]]
// CHECK:         end_kernel
// CHECK:       ^bb2:
// CHECK:         test_inst ins %[[SCC0]]
// CHECK:         end_kernel
amdgcn.kernel @cross_block_scc {
  %scc = alloca : !amdgcn.scc
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %cond = func.call @rand() : () -> i1
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc0, %scc1 : (!amdgcn.scc, !amdgcn.scc) -> ()
  end_kernel
^bb2:
  test_inst ins %scc0 : (!amdgcn.scc) -> ()
  end_kernel
}

// -----

// Diamond CFG: SCC defined in entry, clobbered in both branches, used in both.
// Each branch independently promotes scc0 before its own clobber.

func.func private @rand() -> i1
// CHECK-LABEL: amdgcn.kernel @diamond_both_clobber {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         cf.cond_br
// CHECK:       ^bb1:
// CHECK:         %[[PROMO1:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED1:.*]] = lsir.copy %[[PROMO1]], %[[SCC0]]
// CHECK:         %[[SCC1:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         test_inst ins %[[PROMOTED1]], %[[SCC1]]
// CHECK:       ^bb2:
// CHECK:         %[[PROMO2:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[PROMOTED2:.*]] = lsir.copy %[[PROMO2]], %[[SCC0]]
// CHECK:         %[[SCC2:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         test_inst ins %[[PROMOTED2]], %[[SCC2]]
amdgcn.kernel @diamond_both_clobber {
  %scc = alloca : !amdgcn.scc
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %d = alloca : !amdgcn.sgpr
  %cond = func.call @rand() : () -> i1
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc0, %scc1 : (!amdgcn.scc, !amdgcn.scc) -> ()
  cf.br ^merge
^bb2:
  %scc2 = s_cmp_eq_i32 outs(%scc) ins(%a, %d) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  test_inst ins %scc0, %scc2 : (!amdgcn.scc, !amdgcn.scc) -> ()
  cf.br ^merge
^merge:
  end_kernel
}

// -----

// Multiple special register types live simultaneously: SCC and VCC are
// independently tracked and promoted without interfering.

// CHECK-LABEL: amdgcn.kernel @scc_and_vcc_simultaneous {
// CHECK:         %[[SCC_ALLOC:.*]] = alloca : !amdgcn.scc
// CHECK:         %[[SCC0:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         %[[VCC0:.*]] = v_cmp_eq_i32
// CHECK:         %[[SCC_PROMO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[SCC_PROMOTED:.*]] = lsir.copy %[[SCC_PROMO]], %[[SCC0]]
// CHECK:         %[[SCC1:.*]] = s_cmp_eq_i32 outs(%[[SCC_ALLOC]])
// CHECK:         %[[VCC_PROMO_LO:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VCC_PROMO_HI:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VCC_PROMO:.*]] = make_register_range %[[VCC_PROMO_LO]], %[[VCC_PROMO_HI]]
// CHECK:         %[[VCC_PROMOTED:.*]] = lsir.copy %[[VCC_PROMO]], %[[VCC0]]
// CHECK:         %[[VCC1:.*]] = v_cmp_eq_i32
// CHECK:         test_inst ins %[[SCC_PROMOTED]], %[[VCC_PROMOTED]], %[[SCC1]], %[[VCC1]]
// CHECK:         end_kernel
amdgcn.kernel @scc_and_vcc_simultaneous {
  %scc = alloca : !amdgcn.scc
  %vcc_lo = alloca : !amdgcn.vcc_lo
  %vcc_hi = alloca : !amdgcn.vcc_hi
  %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
  %a = alloca : !amdgcn.sgpr
  %b = alloca : !amdgcn.sgpr
  %c = alloca : !amdgcn.sgpr
  %v1 = alloca : !amdgcn.vgpr
  %v2 = alloca : !amdgcn.vgpr
  %scc0 = s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %vcc0 = v_cmp_eq_i32 outs(%vcc) ins(%a, %v1) : outs(!amdgcn.vcc) ins(!amdgcn.sgpr, !amdgcn.vgpr)
  %scc1 = s_cmp_eq_i32 outs(%scc) ins(%a, %c) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  %vcc1 = v_cmp_eq_i32 outs(%vcc) ins(%a, %v2) : outs(!amdgcn.vcc) ins(!amdgcn.sgpr, !amdgcn.vgpr)
  test_inst ins %scc0, %vcc0, %scc1, %vcc1 : (!amdgcn.scc, !amdgcn.vcc, !amdgcn.scc, !amdgcn.vcc) -> ()
  end_kernel
}
