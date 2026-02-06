// RUN: aster-opt %s --amdgcn-legalize-operands --split-input-file | FileCheck %s

// Test: two non-inline literal constants -> true_value materialized into sgpr.
// Both 544 and 1632 are outside the inline range [-16, 64].

// CHECK-LABEL: kernel @dual_literal_select
// CHECK:         lsir.cmpi
// CHECK:         %[[OUT:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[MOV:.*]] = sop1 s_mov_b32 outs %[[OUT]] ins %{{.*}} : !amdgcn.sgpr, i32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %[[MOV]], %{{.*}} : !amdgcn.sgpr, i1, !amdgcn.sgpr, i32
amdgcn.module @dual_literal_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @dual_literal_select {
    %c0 = arith.constant 0 : i32
    %c544 = arith.constant 544 : i32
    %c1632 = arith.constant 1632 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %cmp = lsir.cmpi i32 eq %s0, %c0 : !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %c544, %c1632 : !amdgcn.sgpr, i1, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: one inline, one non-inline -> no transformation needed.
// 10 is in [-16, 64], so only one literal.

// CHECK-LABEL: kernel @one_inline_select
// CHECK-NOT:     sop1 s_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.sgpr, i1, i32, i32
amdgcn.module @one_inline_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @one_inline_select {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c200 = arith.constant 200 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %cmp = lsir.cmpi i32 eq %s0, %c0 : !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %c10, %c200 : !amdgcn.sgpr, i1, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: both inline constants -> no transformation needed.
// 0 and 10 are both in [-16, 64].

// CHECK-LABEL: kernel @both_inline_select
// CHECK-NOT:     sop1 s_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !amdgcn.sgpr, i1, i32, i32
amdgcn.module @both_inline_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @both_inline_select {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c20 = arith.constant 20 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %cmp = lsir.cmpi i32 eq %s0, %c0 : !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %c10, %c20 : !amdgcn.sgpr, i1, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: non-constant operands -> no transformation.
// When operands are register values (not arith.constant), pass does nothing.

// CHECK-LABEL: kernel @non_constant_select
// CHECK:         %[[A:.*]] = sop1 s_mov_b32
// CHECK:         %[[B:.*]] = sop1 s_mov_b32
// CHECK:         lsir.select %{{.*}}, %{{.*}}, %[[A]], %[[B]] : !amdgcn.sgpr, i1, !amdgcn.sgpr, !amdgcn.sgpr
amdgcn.module @non_constant_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @non_constant_select {
    %c0 = arith.constant 0 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %s2 = alloca : !amdgcn.sgpr
    %a = sop1 s_mov_b32 outs %s1 ins %c0 : !amdgcn.sgpr, i32
    %b = sop1 s_mov_b32 outs %s2 ins %c0 : !amdgcn.sgpr, i32
    %cmp = lsir.cmpi i32 eq %s0, %c0 : !amdgcn.sgpr, i32
    %sel = lsir.select %s0, %cmp, %a, %b : !amdgcn.sgpr, i1, !amdgcn.sgpr, !amdgcn.sgpr
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}

// -----

// Test: boundary values at inline constant edges.
// -16 and 64 are the boundary inline constants; -17 and 65 are not.

// CHECK-LABEL: kernel @boundary_inline_select
// CHECK-NOT:     sop1 s_mov_b32
// CHECK:         lsir.select {{.*}} : !amdgcn.sgpr, i1, i32, i32
amdgcn.module @boundary_inline_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @boundary_inline_select {
    %c0 = arith.constant 0 : i32
    %cn16 = arith.constant -16 : i32
    %c64 = arith.constant 64 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %cmp = lsir.cmpi i32 eq %s0, %c0 : !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %cn16, %c64 : !amdgcn.sgpr, i1, i32, i32
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
// CHECK:         sop1 s_mov_b32 outs %[[OUT]]
// CHECK:         lsir.select {{.*}} : !amdgcn.sgpr, i1, !amdgcn.sgpr, i32
amdgcn.module @boundary_non_inline_mod target = <gfx942> isa = <cdna3> {
  amdgcn.kernel @boundary_non_inline_select {
    %c0 = arith.constant 0 : i32
    %cn17 = arith.constant -17 : i32
    %c65 = arith.constant 65 : i32
    %s0 = alloca : !amdgcn.sgpr
    %s1 = alloca : !amdgcn.sgpr
    %cmp = lsir.cmpi i32 eq %s0, %c0 : !amdgcn.sgpr, i32
    %sel = lsir.select %s1, %cmp, %cn17, %c65 : !amdgcn.sgpr, i1, i32, i32
    test_inst ins %sel : (!amdgcn.sgpr) -> ()
    end_kernel
  }
}
