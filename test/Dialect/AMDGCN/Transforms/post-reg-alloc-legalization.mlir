// RUN: aster-opt %s --amdgcn-post-reg-alloc-legalization --split-input-file | FileCheck %s

// Positive case: SGPR->SGPR copy (lsir.copy target, source) expands to
// s_mov_b32. The first operand is the destination (target), the second is
// the source in DPS style.

// CHECK-LABEL: amdgcn.kernel @sgpr_copy_expands {
// CHECK-DAG:     %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:     %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:         s_mov_b32 outs(%[[S0]]) ins(%[[S1]]) : outs(!amdgcn.sgpr<0>) ins(!amdgcn.sgpr<1>)
// CHECK:         test_inst ins %[[S0]] : (!amdgcn.sgpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @sgpr_copy_expands {
  %tgt = alloca : !amdgcn.sgpr<0>
  %src = alloca : !amdgcn.sgpr<1>
  test_inst outs %src : (!amdgcn.sgpr<1>) -> ()
  // lsir.copy target, source — tgt is the destination register.
  lsir.copy %tgt, %src : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
  test_inst ins %tgt : (!amdgcn.sgpr<0>) -> ()
  end_kernel
}

// -----

// Positive case: VGPR->VGPR copy expands to v_mov_b32.

// CHECK-LABEL: amdgcn.kernel @vgpr_copy_expands {
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:         v_mov_b32 outs(%[[V0]]) ins(%[[V1]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<1>)
// CHECK:         test_inst ins %[[V0]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @vgpr_copy_expands {
  %tgt = alloca : !amdgcn.vgpr<0>
  %src = alloca : !amdgcn.vgpr<1>
  test_inst outs %src : (!amdgcn.vgpr<1>) -> ()
  lsir.copy %tgt, %src : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
  test_inst ins %tgt : (!amdgcn.vgpr<0>) -> ()
  end_kernel
}

// -----

// Positive case: AGPR->AGPR copy expands to v_accvgpr_mov_b32.

// CHECK-LABEL: amdgcn.kernel @agpr_copy_expands {
// CHECK-DAG:     %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK-DAG:     %[[A1:.*]] = alloca : !amdgcn.agpr<1>
// CHECK:         v_accvgpr_mov_b32 outs(%[[A0]]) ins(%[[A1]]) : outs(!amdgcn.agpr<0>) ins(!amdgcn.agpr<1>)
// CHECK:         test_inst ins %[[A0]] : (!amdgcn.agpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @agpr_copy_expands {
  %tgt = alloca : !amdgcn.agpr<0>
  %src = alloca : !amdgcn.agpr<1>
  test_inst outs %src : (!amdgcn.agpr<1>) -> ()
  lsir.copy %tgt, %src : !amdgcn.agpr<0>, !amdgcn.agpr<1>
  test_inst ins %tgt : (!amdgcn.agpr<0>) -> ()
  end_kernel
}

// -----

// Negative case: redundant v_mov_b32 (src == dst physical reg) is erased.
// CHECK-LABEL: kernel @redundant_v_mov_erased {
// CHECK-NOT:     v_mov_b32
amdgcn.module @redundant_v_mov_erased_mod target = <gfx942> {
  amdgcn.kernel @redundant_v_mov_erased {
    %v0 = alloca : !amdgcn.vgpr<0>
    amdgcn.v_mov_b32 outs(%v0) ins(%v0) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.vgpr<0>)
    end_kernel
  }
}

// -----

// Negative case: redundant s_mov_b32 (src == dst physical reg) is erased.
// CHECK-LABEL: kernel @redundant_s_mov_erased {
// CHECK-NOT:     s_mov_b32
amdgcn.module @redundant_s_mov_erased_mod target = <gfx942> {
  amdgcn.kernel @redundant_s_mov_erased {
    %s0 = alloca : !amdgcn.sgpr<0>
    amdgcn.s_mov_b32 outs(%s0) ins(%s0) : outs(!amdgcn.sgpr<0>) ins(!amdgcn.sgpr<0>)
    end_kernel
  }
}

// -----

// Negative case: redundant v_accvgpr_mov_b32 (src == dst) is erased.
// CHECK-LABEL: kernel @redundant_v_accvgpr_mov_erased {
// CHECK-NOT:     v_accvgpr_mov_b32
amdgcn.module @redundant_v_accvgpr_mov_erased_mod target = <gfx942> {
  amdgcn.kernel @redundant_v_accvgpr_mov_erased {
    %a0 = alloca : !amdgcn.agpr<0>
    amdgcn.v_accvgpr_mov_b32 outs(%a0) ins(%a0) : outs(!amdgcn.agpr<0>) ins(!amdgcn.agpr<0>)
    end_kernel
  }
}

// -----

// Negative case: lsir.copy with unallocated operands is not expanded because
// the operands still need register allocation.

// CHECK-LABEL: amdgcn.kernel @unallocated_copy_not_expanded {
// CHECK-DAG:     %[[TGT:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK-DAG:     %[[SRC:.*]] = alloca : !amdgcn.vgpr<?>
// CHECK:         lsir.copy %[[TGT]], %[[SRC]] : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
// CHECK:         end_kernel
amdgcn.kernel @unallocated_copy_not_expanded {
  %tgt = alloca : !amdgcn.vgpr<?>
  %src = alloca : !amdgcn.vgpr<?>
  lsir.copy %tgt, %src : !amdgcn.vgpr<?>, !amdgcn.vgpr<?>
  end_kernel
}

// -----

// Negative case: VGPR->SGPR cross-class copy is not expanded because hardware
// cannot perform this move directly.

// CHECK-LABEL: amdgcn.kernel @vgpr_to_sgpr_copy_not_expanded {
// CHECK-DAG:     %[[DST:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:     %[[SRC:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:         lsir.copy %[[DST]], %[[SRC]] : !amdgcn.sgpr<0>, !amdgcn.vgpr<0>
// CHECK:         end_kernel
amdgcn.kernel @vgpr_to_sgpr_copy_not_expanded {
  %dst = alloca : !amdgcn.sgpr<0>
  %src = alloca : !amdgcn.vgpr<0>
  lsir.copy %dst, %src : !amdgcn.sgpr<0>, !amdgcn.vgpr<0>
  end_kernel
}

// -----

// Positive case: lsir.copy of a two-register VGPR range expands to two
// v_mov_b32 instructions, one per register pair.

// CHECK-LABEL: amdgcn.kernel @vgpr_range_copy_expands {
// CHECK-DAG:     %[[T0:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:     %[[T1:.*]] = alloca : !amdgcn.vgpr<3>
// CHECK-DAG:     %[[S0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[S1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:         v_mov_b32 outs(%[[T0]]) ins(%[[S0]]) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>)
// CHECK:         v_mov_b32 outs(%[[T1]]) ins(%[[S1]]) : outs(!amdgcn.vgpr<3>) ins(!amdgcn.vgpr<1>)
// CHECK:         test_inst ins %[[T0]], %[[T1]] : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @vgpr_range_copy_expands {
  %tgt0 = alloca : !amdgcn.vgpr<2>
  %tgt1 = alloca : !amdgcn.vgpr<3>
  %src0 = alloca : !amdgcn.vgpr<0>
  %src1 = alloca : !amdgcn.vgpr<1>
  test_inst outs %src0, %src1 : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
  %tgt = make_register_range %tgt0, %tgt1 : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
  %src = make_register_range %src0, %src1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
  lsir.copy %tgt, %src : !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[0 : 2]>
  test_inst ins %tgt0, %tgt1 : (!amdgcn.vgpr<2>, !amdgcn.vgpr<3>) -> ()
  end_kernel
}

// -----

// Positive case: SGPR-source to VGPR-target copy expands to v_mov_b32.
// Hardware supports reading an SGPR as the source of a VGPR write.

// CHECK-LABEL: amdgcn.kernel @sgpr_to_vgpr_copy_expands {
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:         v_mov_b32 outs(%[[V0]]) ins(%[[S0]]) : outs(!amdgcn.vgpr<0>) ins(!amdgcn.sgpr<0>)
// CHECK:         test_inst ins %[[V0]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         end_kernel
amdgcn.kernel @sgpr_to_vgpr_copy_expands {
  %tgt = alloca : !amdgcn.vgpr<0>
  %src = alloca : !amdgcn.sgpr<0>
  test_inst outs %src : (!amdgcn.sgpr<0>) -> ()
  lsir.copy %tgt, %src : !amdgcn.vgpr<0>, !amdgcn.sgpr<0>
  test_inst ins %tgt : (!amdgcn.vgpr<0>) -> ()
  end_kernel
}

// -----

// Negative case: VGPR-to-AGPR copy is not expanded because hardware requires
// an explicit v_accvgpr_write_b32 for this transfer.

// CHECK-LABEL: amdgcn.kernel @vgpr_to_agpr_copy_not_expanded {
// CHECK-DAG:     %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:         lsir.copy %[[A0]], %[[V0]] : !amdgcn.agpr<0>, !amdgcn.vgpr<0>
// CHECK:         end_kernel
amdgcn.kernel @vgpr_to_agpr_copy_not_expanded {
  %tgt = alloca : !amdgcn.agpr<0>
  %src = alloca : !amdgcn.vgpr<0>
  test_inst outs %src : (!amdgcn.vgpr<0>) -> ()
  lsir.copy %tgt, %src : !amdgcn.agpr<0>, !amdgcn.vgpr<0>
  end_kernel
}

// -----

// Negative case: AGPR-to-VGPR copy is not expanded because hardware requires
// an explicit v_accvgpr_read_b32 for this transfer.

// CHECK-LABEL: amdgcn.kernel @agpr_to_vgpr_copy_not_expanded {
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[A0:.*]] = alloca : !amdgcn.agpr<0>
// CHECK:         lsir.copy %[[V0]], %[[A0]] : !amdgcn.vgpr<0>, !amdgcn.agpr<0>
// CHECK:         end_kernel
amdgcn.kernel @agpr_to_vgpr_copy_not_expanded {
  %tgt = alloca : !amdgcn.vgpr<0>
  %src = alloca : !amdgcn.agpr<0>
  test_inst outs %src : (!amdgcn.agpr<0>) -> ()
  lsir.copy %tgt, %src : !amdgcn.vgpr<0>, !amdgcn.agpr<0>
  end_kernel
}
