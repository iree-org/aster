// RUN: aster-opt %s --amdgcn-legalize-operands --split-input-file | FileCheck %s

// Test: two non-inline literal constants -> true_value materialized into sgpr.
// Both 544 and 1632 are outside the inline range [-16, 64].

// CHECK-LABEL: kernel @dual_literal_select
// CHECK:         lsir.cmpi
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[MOV:.*]] = s_mov_b32 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.sgpr) ins(i32)
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %[[MOV]], %{{.*}} : !amdgcn.sgpr, !amdgcn.scc, !amdgcn.sgpr, i32
amdgcn.module @dual_literal_mod target = <gfx942> {
  amdgcn.kernel @dual_literal_select {
    %c0 = arith.constant 0 : i32
    %c544 = arith.constant 544 : i32
    %c1632 = arith.constant 1632 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = lsir.cmpi i32 eq %scc, %s0, %c0 : !amdgcn.scc, !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %c544, %c1632 : !amdgcn.sgpr, !amdgcn.scc, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: one inline, one non-inline -> no transformation needed.
// 10 is in [-16, 64], so only one literal.

// CHECK-LABEL: kernel @one_inline_select
// CHECK-NOT:     s_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.sgpr, !amdgcn.scc, i32, i32
amdgcn.module @one_inline_mod target = <gfx942> {
  amdgcn.kernel @one_inline_select {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c200 = arith.constant 200 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = lsir.cmpi i32 eq %scc, %s0, %c0 : !amdgcn.scc, !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %c10, %c200 : !amdgcn.sgpr, !amdgcn.scc, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: both inline constants -> no transformation needed.
// 0 and 10 are both in [-16, 64].

// CHECK-LABEL: kernel @both_inline_select
// CHECK-NOT:     s_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.sgpr, !amdgcn.scc, i32, i32
amdgcn.module @both_inline_mod target = <gfx942> {
  amdgcn.kernel @both_inline_select {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c20 = arith.constant 20 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = lsir.cmpi i32 eq %scc, %s0, %c0 : !amdgcn.scc, !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %c10, %c20 : !amdgcn.sgpr, !amdgcn.scc, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: non-constant operands -> no transformation.
// When operands are register values (not arith.constant), pass does nothing.

// CHECK-LABEL: kernel @non_constant_select
// CHECK:         %[[A:.*]] = s_mov_b32
// CHECK:         %[[B:.*]] = s_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %[[A]], %[[B]] : !amdgcn.sgpr, !amdgcn.scc, !amdgcn.sgpr, !amdgcn.sgpr
amdgcn.module @non_constant_mod target = <gfx942> {
  amdgcn.kernel @non_constant_select {
    %c0 = arith.constant 0 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %s2 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %a = s_mov_b32 outs(%s1) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    %b = s_mov_b32 outs(%s2) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    %cmp = lsir.cmpi i32 eq %scc, %s0, %c0 : !amdgcn.scc, !amdgcn.sgpr, i32
    %sel = lsir.select %s0, %cmp, %a, %b : !amdgcn.sgpr, !amdgcn.scc, !amdgcn.sgpr, !amdgcn.sgpr
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: boundary values at inline constant edges.
// -16 and 64 are the boundary inline constants; -17 and 65 are not.

// CHECK-LABEL: kernel @boundary_inline_select
// CHECK-NOT:     s_mov_b32
// CHECK:         lsir.select {{.*}} : !amdgcn.sgpr, !amdgcn.scc, i32, i32
amdgcn.module @boundary_inline_mod target = <gfx942> {
  amdgcn.kernel @boundary_inline_select {
    %c0 = arith.constant 0 : i32
    %cn16 = arith.constant -16 : i32
    %c64 = arith.constant 64 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = lsir.cmpi i32 eq %scc, %s0, %c0 : !amdgcn.scc, !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %cn16, %c64 : !amdgcn.sgpr, !amdgcn.scc, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: boundary values just outside inline range -> needs legalization.
// -17 and 65 are both outside [-16, 64].

// CHECK-LABEL: kernel @boundary_non_inline_select
// CHECK:         lsir.cmpi
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.sgpr
// CHECK:         s_mov_b32 outs(%[[OUT]])
// CHECK:         lsir.select {{.*}} : !amdgcn.sgpr, !amdgcn.scc, !amdgcn.sgpr, i32
amdgcn.module @boundary_non_inline_mod target = <gfx942> {
  amdgcn.kernel @boundary_non_inline_select {
    %c0 = arith.constant 0 : i32
    %cn17 = arith.constant -17 : i32
    %c65 = arith.constant 65 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = lsir.cmpi i32 eq %scc, %s0, %c0 : !amdgcn.scc, !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %cn17, %c65 : !amdgcn.sgpr, !amdgcn.scc, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: VCC condition with non-inline constant true_value -> materialized into
// a new vgpr via v_mov_b32. 544 is outside the inline range [-16, 64].

// CHECK-LABEL: kernel @vcndmask_non_inline_true
// CHECK:         lsir.cmpi
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[MOV:.*]] = v_mov_b32 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.vgpr) ins(i32)
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %[[MOV]], %{{.*}} : !amdgcn.vgpr, !amdgcn.vcc, !amdgcn.vgpr, i32
amdgcn.module @vcndmask_non_inline_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_non_inline_true {
    %c0 = arith.constant 0 : i32
    %c544 = arith.constant 544 : i32
    %c1632 = arith.constant 1632 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %vcc_lo = alloca : !amdgcn.vcc_lo
    %vcc_hi = alloca : !amdgcn.vcc_hi
    %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
    %cmp = lsir.cmpi i32 eq %vcc, %v0, %c0 : !amdgcn.vcc, !amdgcn.vgpr, i32
    %sel = lsir.select %v1, %cmp, %c544, %c1632 : !amdgcn.vgpr, !amdgcn.vcc, i32, i32
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: VCC condition with inline constant true_value -> no transformation.
// 10 is in [-16, 64], so VOP3 can handle it directly.

// CHECK-LABEL: kernel @vcndmask_inline_true
// CHECK-NOT:     v_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.vgpr, !amdgcn.vcc, i32, i32
amdgcn.module @vcndmask_inline_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_inline_true {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c200 = arith.constant 200 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %vcc_lo = alloca : !amdgcn.vcc_lo
    %vcc_hi = alloca : !amdgcn.vcc_hi
    %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
    %cmp = lsir.cmpi i32 eq %vcc, %v0, %c0 : !amdgcn.vcc, !amdgcn.vgpr, i32
    %sel = lsir.select %v1, %cmp, %c10, %c200 : !amdgcn.vgpr, !amdgcn.vcc, i32, i32
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: VCC condition with VGPR true_value -> no transformation.
// true_value is already a VGPR, satisfying the VOP2 src1 constraint.

// CHECK-LABEL: kernel @vcndmask_vgpr_true
// CHECK:         %[[VTRUE:.*]] = v_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %[[VTRUE]], %{{.*}} : !amdgcn.vgpr, !amdgcn.vcc, !amdgcn.vgpr, i32
amdgcn.module @vcndmask_vgpr_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_vgpr_true {
    %c0 = arith.constant 0 : i32
    %c200 = arith.constant 200 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %v2 = alloca : !amdgcn.vgpr
    %vcc_lo = alloca : !amdgcn.vcc_lo
    %vcc_hi = alloca : !amdgcn.vcc_hi
    %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
    %vtrue = v_mov_b32 outs(%v2) ins(%c0) : outs(!amdgcn.vgpr) ins(i32)
    %cmp = lsir.cmpi i32 eq %vcc, %v0, %c0 : !amdgcn.vcc, !amdgcn.vgpr, i32
    %sel = lsir.select %v1, %cmp, %vtrue, %c200 : !amdgcn.vgpr, !amdgcn.vcc, !amdgcn.vgpr, i32
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: VCC condition, non-inline true_value but inline false_value.
// VCndmaskB32LegalizePattern only guards on true_value; false_value being
// inline does not prevent materialization of true_value into a VGPR.

// CHECK-LABEL: kernel @vcndmask_non_inline_true_inline_false
// CHECK:         lsir.cmpi
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[MOV:.*]] = v_mov_b32 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.vgpr) ins(i32)
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %[[MOV]], %{{.*}} : !amdgcn.vgpr, !amdgcn.vcc, !amdgcn.vgpr, i32
amdgcn.module @vcndmask_non_inline_true_inline_false_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_non_inline_true_inline_false {
    %c0 = arith.constant 0 : i32
    %c544 = arith.constant 544 : i32
    %c10 = arith.constant 10 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %vcc_lo = alloca : !amdgcn.vcc_lo
    %vcc_hi = alloca : !amdgcn.vcc_hi
    %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
    %cmp = lsir.cmpi i32 eq %vcc, %v0, %c0 : !amdgcn.vcc, !amdgcn.vgpr, i32
    %sel = lsir.select %v1, %cmp, %c544, %c10 : !amdgcn.vgpr, !amdgcn.vcc, i32, i32
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: SCC condition, non-inline true_value but inline false_value.
// SOP2 allows one literal, so no materialization is needed when only
// one operand is non-inline.

// CHECK-LABEL: kernel @scc_non_inline_true_inline_false
// CHECK-NOT:     s_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.sgpr, !amdgcn.scc, i32, i32
amdgcn.module @scc_non_inline_true_inline_false_mod target = <gfx942> {
  amdgcn.kernel @scc_non_inline_true_inline_false {
    %c0 = arith.constant 0 : i32
    %c544 = arith.constant 544 : i32
    %c10 = arith.constant 10 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = lsir.cmpi i32 eq %scc, %s0, %c0 : !amdgcn.scc, !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %c544, %c10 : !amdgcn.sgpr, !amdgcn.scc, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: s_cselect_b32 with two non-inline literals -> src0 materialized into
// an SGPR via alloca + s_mov_b32. Both 544 and 1632 are outside [-16, 64].

// CHECK-LABEL: kernel @scselect_b32_dual_literal
// CHECK:         s_cmp_eq_i32
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[MOV:.*]] = s_mov_b32 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.sgpr) ins(i32)
// CHECK:         s_cselect_b32 outs(%{{.*}}) ins(%[[MOV]], %{{.*}}, %{{.*}}) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr, i32, !amdgcn.scc)
amdgcn.module @scselect_b32_dual_literal_mod target = <gfx942> {
  amdgcn.kernel @scselect_b32_dual_literal {
    %c0 = arith.constant 0 : i32
    %c544 = arith.constant 544 : i32
    %c1632 = arith.constant 1632 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = amdgcn.s_cmp_eq_i32 outs(%scc) ins(%s0, %c0) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, i32)
    %sel = amdgcn.s_cselect_b32 outs(%s1) ins(%c544, %c1632, %cmp) : outs(!amdgcn.sgpr) ins(i32, i32, !amdgcn.scc)
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: s_cselect_b32 with one inline, one non-inline -> no transformation.
// 10 is in [-16, 64] so only one literal, which SOP2 can encode directly.

// CHECK-LABEL: kernel @scselect_b32_one_inline
// CHECK-NOT:     s_mov_b32
// CHECK:         s_cselect_b32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}) : outs(!amdgcn.sgpr) ins(i32, i32, !amdgcn.scc)
amdgcn.module @scselect_b32_one_inline_mod target = <gfx942> {
  amdgcn.kernel @scselect_b32_one_inline {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c200 = arith.constant 200 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = amdgcn.s_cmp_eq_i32 outs(%scc) ins(%s0, %c0) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, i32)
    %sel = amdgcn.s_cselect_b32 outs(%s1) ins(%c10, %c200, %cmp) : outs(!amdgcn.sgpr) ins(i32, i32, !amdgcn.scc)
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: s_cselect_b64 with two non-inline i64 literals -> src0 materialized
// into a two-SGPR range via alloca + s_mov_b64.

// CHECK-LABEL: kernel @scselect_b64_dual_literal
// CHECK:         s_cmp_eq_i32
// CHECK:         %[[OUT:.*]] = make_register_range
// CHECK:         %[[MOV:.*]] = s_mov_b64 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.sgpr<[? + 2]>) ins(i64)
// CHECK:         s_cselect_b64 outs(%{{.*}}) ins(%[[MOV]], %{{.*}}, %{{.*}}) : outs(!amdgcn.sgpr<[? + 2]>) ins(!amdgcn.sgpr<[? + 2]>, i64, !amdgcn.scc)
amdgcn.module @scselect_b64_dual_literal_mod target = <gfx942> {
  amdgcn.kernel @scselect_b64_dual_literal {
    %c0 = arith.constant 0 : i32
    %c544 = arith.constant 544 : i64
    %c1632 = arith.constant 1632 : i64
    %s0 = alloca : !amdgcn.sgpr
    %s2 = alloca : !amdgcn.sgpr
    %s3 = alloca : !amdgcn.sgpr
    %ss1 = make_register_range %s2, %s3 : !amdgcn.sgpr, !amdgcn.sgpr
    %scc = alloca : !amdgcn.scc
    %cmp = amdgcn.s_cmp_eq_i32 outs(%scc) ins(%s0, %c0) : outs(!amdgcn.scc) ins(!amdgcn.sgpr, i32)
    %sel = amdgcn.s_cselect_b64 outs(%ss1) ins(%c544, %c1632, %cmp) : outs(!amdgcn.sgpr<[? + 2]>) ins(i64, i64, !amdgcn.scc)
    test_inst ins %sel : (!amdgcn.sgpr<[? + 2]>) -> ()
    end_kernel
  }
}

// -----

// Test: v_cndmask_b32 with non-inline src1 -> src1 materialized into a VGPR
// via alloca + v_mov_b32. 544 is outside [-16, 64].

// CHECK-LABEL: kernel @vcndmask_b32_non_inline_src1
// CHECK:         v_cmp_eq_i32
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[MOV:.*]] = v_mov_b32 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.vgpr) ins(i32)
// CHECK:         v_cndmask_b32 outs(%{{.*}}) ins(%{{.*}}, %[[MOV]], %{{.*}}) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr, !amdgcn.vcc)
amdgcn.module @vcndmask_b32_non_inline_src1_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_b32_non_inline_src1 {
    %c0 = arith.constant 0 : i32
    %c544 = arith.constant 544 : i32
    %c1632 = arith.constant 1632 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %vcc_lo = alloca : !amdgcn.vcc_lo
    %vcc_hi = alloca : !amdgcn.vcc_hi
    %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
    %cmp = amdgcn.v_cmp_eq_i32 outs(%vcc) ins(%v0, %c0) : outs(!amdgcn.vcc) ins(!amdgcn.vgpr, i32)
    %sel = amdgcn.v_cndmask_b32 outs(%v1) ins(%c1632, %c544, %cmp) : outs(!amdgcn.vgpr) ins(i32, i32, !amdgcn.vcc)
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: v_cndmask_b32 with inline src1 -> no transformation.
// 10 is in [-16, 64] so VOP3 can encode it directly.

// CHECK-LABEL: kernel @vcndmask_b32_inline_src1
// CHECK-NOT:     v_mov_b32
// CHECK:         v_cndmask_b32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}) : outs(!amdgcn.vgpr) ins(i32, i32, !amdgcn.vcc)
amdgcn.module @vcndmask_b32_inline_src1_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_b32_inline_src1 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c200 = arith.constant 200 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %vcc_lo = alloca : !amdgcn.vcc_lo
    %vcc_hi = alloca : !amdgcn.vcc_hi
    %vcc = make_register_range %vcc_lo, %vcc_hi : !amdgcn.vcc_lo, !amdgcn.vcc_hi
    %cmp = amdgcn.v_cmp_eq_i32 outs(%vcc) ins(%v0, %c0) : outs(!amdgcn.vcc) ins(!amdgcn.vgpr, i32)
    %sel = amdgcn.v_cndmask_b32 outs(%v1) ins(%c200, %c10, %cmp) : outs(!amdgcn.vgpr) ins(i32, i32, !amdgcn.vcc)
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: lsir.select with SGPR (non-SCC, non-VCC) condition + non-inline
// true_value. SGPR conditions lower to v_cndmask_b32 (VALU), so the true
// value must be materialized into a VGPR via v_mov_b32, not s_mov_b32.

// CHECK-LABEL: kernel @sgpr_cond_select_non_inline_true
// CHECK:         %{{.*}} = alloca : !amdgcn.vgpr
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[MOV:.*]] = v_mov_b32 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.vgpr) ins(i32)
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %[[MOV]], %{{.*}} : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr, i32
// CHECK-NOT:     s_mov_b32
amdgcn.module @sgpr_cond_select_non_inline_mod target = <gfx942> {
  amdgcn.kernel @sgpr_cond_select_non_inline_true {
    %c544 = arith.constant 544 : i32
    %c1632 = arith.constant 1632 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %s0 = alloca : !amdgcn.sgpr
    %sel = lsir.select %v1, %s0, %c544, %c1632 : !amdgcn.vgpr, !amdgcn.sgpr, i32, i32
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: lsir.select with SGPR condition + inline true_value -> no
// transformation. 10 is in [-16, 64], so VOP3 can encode it directly.

// CHECK-LABEL: kernel @sgpr_cond_select_inline_true
// CHECK-NOT:     v_mov_b32
// CHECK-NOT:     s_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.vgpr, !amdgcn.sgpr, i32, i32
amdgcn.module @sgpr_cond_select_inline_mod target = <gfx942> {
  amdgcn.kernel @sgpr_cond_select_inline_true {
    %c10 = arith.constant 10 : i32
    %c200 = arith.constant 200 : i32
    %v1 = alloca : !amdgcn.vgpr
    %s0 = alloca : !amdgcn.sgpr
    %sel = lsir.select %v1, %s0, %c10, %c200 : !amdgcn.vgpr, !amdgcn.sgpr, i32, i32
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: v_cndmask_b32 with a two-SGPR condition (VOP3 mode, non-VCC) and
// non-inline src1. The condition is not SCC, so VGPRSelectInstLegalizePattern
// fires and materializes src1 into a VGPR via v_mov_b32.

// CHECK-LABEL: kernel @vcndmask_twosgpr_cond_non_inline_src1
// CHECK:         %{{.*}} = alloca : !amdgcn.vgpr
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.vgpr
// CHECK:         %[[MOV:.*]] = v_mov_b32 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.vgpr) ins(i32)
// CHECK:         v_cndmask_b32 outs(%{{.*}}) ins(%{{.*}}, %[[MOV]], %{{.*}}) : outs(!amdgcn.vgpr) ins(i32, !amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>)
amdgcn.module @vcndmask_twosgpr_cond_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_twosgpr_cond_non_inline_src1 {
    %c544 = arith.constant 544 : i32
    %c1632 = arith.constant 1632 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %cond = make_register_range %s0, %s1 : !amdgcn.sgpr, !amdgcn.sgpr
    test_inst ins %cond : (!amdgcn.sgpr<[? + 2]>) -> ()
    %sel = amdgcn.v_cndmask_b32 outs(%v1) ins(%c1632, %c544, %cond) : outs(!amdgcn.vgpr) ins(i32, i32, !amdgcn.sgpr<[? + 2]>)
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}

// -----

// Test: v_cndmask_b32 with a two-SGPR condition + inline src1 -> no
// transformation. 10 is in [-16, 64], so VOP3 can encode it directly.

// CHECK-LABEL: kernel @vcndmask_twosgpr_cond_inline_src1
// CHECK-NOT:     v_mov_b32
// CHECK:         v_cndmask_b32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}) : outs(!amdgcn.vgpr) ins(i32, i32, !amdgcn.sgpr<[? + 2]>)
amdgcn.module @vcndmask_twosgpr_cond_inline_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_twosgpr_cond_inline_src1 {
    %c10 = arith.constant 10 : i32
    %c200 = arith.constant 200 : i32
    %v0 = alloca : !amdgcn.vgpr
    %v1 = alloca : !amdgcn.vgpr
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %cond = make_register_range %s0, %s1 : !amdgcn.sgpr, !amdgcn.sgpr
    test_inst ins %cond : (!amdgcn.sgpr<[? + 2]>) -> ()
    %sel = amdgcn.v_cndmask_b32 outs(%v1) ins(%c200, %c10, %cond) : outs(!amdgcn.vgpr) ins(i32, i32, !amdgcn.sgpr<[? + 2]>)
    test_inst ins %sel : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}
