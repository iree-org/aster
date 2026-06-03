// RUN: aster-opt %s --amdgcn-legalize-operands --split-input-file | FileCheck %s

// CHECK-LABEL: kernel @vcndmask_b32_no_literal
// CHECK:         v_cmp_eq_i32
// CHECK:         %[[MOVA:.*]] = v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr) ins(i32)
// CHECK:         %[[MOVB:.*]] = v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr) ins(i32)
// CHECK:         v_cndmask_b32 outs(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vcc)
amdgcn.module @vcndmask_b32_no_literal_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_b32_no_literal {
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

// CHECK-LABEL: kernel @vcndmask_b32_no_literal_src0
// CHECK:         %[[MOV0:.*]] = v_mov_b32 outs(%{{.*}}) ins(%{{.*}}) : outs(!amdgcn.vgpr) ins(i32)
// CHECK:         v_cndmask_b32 outs(%{{.*}}) ins(%[[MOV0]], %{{.*}}, %{{.*}}) : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32, !amdgcn.vcc)
amdgcn.module @vcndmask_b32_no_literal_src0_mod target = <gfx942> {
  amdgcn.kernel @vcndmask_b32_no_literal_src0 {
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

// CHECK-LABEL: kernel @scselect_b32_no_dual_literal
// CHECK:         s_cmp_eq_i32
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[MOV:.*]] = s_mov_b32 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.sgpr) ins(i32)
// CHECK:         s_cselect_b32 outs(%{{.*}}) ins(%[[MOV]], %{{.*}}, %{{.*}}) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr, i32, !amdgcn.scc)
amdgcn.module @scselect_b32_no_dual_literal_mod target = <gfx942> {
  amdgcn.kernel @scselect_b32_no_dual_literal {
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

// s_cselect (SOP2) tolerates one literal, so a single non-inline operand is a no-op.
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

// CHECK-LABEL: kernel @scselect_b64_no_dual_literal
// CHECK:         s_cmp_eq_i32
// CHECK:         %[[OUT:.*]] = make_register_range
// CHECK:         %[[MOV:.*]] = s_mov_b64 outs(%[[OUT]]) ins(%{{.*}}) : outs(!amdgcn.sgpr<[? + 2]>) ins(i64)
// CHECK:         s_cselect_b64 outs(%{{.*}}) ins(%[[MOV]], %{{.*}}, %{{.*}}) : outs(!amdgcn.sgpr<[? + 2]>) ins(!amdgcn.sgpr<[? + 2]>, i64, !amdgcn.scc)
amdgcn.module @scselect_b64_no_dual_literal_mod target = <gfx942> {
  amdgcn.kernel @scselect_b64_no_dual_literal {
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
