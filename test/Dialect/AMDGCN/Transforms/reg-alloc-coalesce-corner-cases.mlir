// RUN: aster-opt %s --amdgcn-reg-alloc | FileCheck %s
// RUN: aster-opt %s --amdgcn-reg-alloc='num-sgprs=4' | FileCheck %s --check-prefix=PRESSURE

amdgcn.module @m target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: kernel @conflicting_fixed_coalesce
  //       CHECK:   %[[BID:.*]] = alloca : !amdgcn.sgpr<2>
  //       CHECK:   %[[TID:.*]] = alloca : !amdgcn.sgpr<1>
  //   first copy coalesces into block_id at sgpr<2>; second uses tid at sgpr<1>:
  //       CHECK:   s_add_u32 {{.*}} ins({{.*}}, %[[BID]]) : {{.*}} ins({{.*}}, !amdgcn.sgpr<2>)
  //       CHECK:   s_add_u32 {{.*}} ins({{.*}}, %[[TID]]) : {{.*}} ins({{.*}}, !amdgcn.sgpr<1>)
  amdgcn.kernel @conflicting_fixed_coalesce attributes {grid_dims = array<i32: 8, 1, 1>} {
    %bid  = amdgcn.alloca : !amdgcn.sgpr<2>
    %tid  = amdgcn.alloca : !amdgcn.sgpr<1>
    %g    = amdgcn.alloca : !amdgcn.sgpr
    %cp1  = lsir.copy %g, %bid : !amdgcn.sgpr, !amdgcn.sgpr<2>
    %cp2  = lsir.copy %g, %tid : !amdgcn.sgpr, !amdgcn.sgpr<1>
    %acc  = amdgcn.alloca : !amdgcn.sgpr
    %cc   = amdgcn.alloca : !amdgcn.scc<0>
    %c0   = arith.constant 0 : i32
    %b    = amdgcn.s_mov_b32 outs(%acc) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    %r    = amdgcn.s_add_u32 outs(%acc, %cc) ins(%b, %cp1) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    %r2   = amdgcn.s_add_u32 outs(%acc, %cc) ins(%r, %cp2) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %r2 : (!amdgcn.sgpr) -> ()
    amdgcn.end_kernel
  }

  // CHECK-LABEL: kernel @fixed_livein_copy
  //       CHECK:   %[[BID:.*]] = alloca : !amdgcn.sgpr<2>
  //       CHECK:   %[[BASE:.*]] = alloca : !amdgcn.sgpr<0>
  //       CHECK:   s_mov_b32 outs(%[[BASE]]) ins({{.*}}) : outs(!amdgcn.sgpr<0>) ins(i32)
  //   block_id must be read from sgpr<2> (where it lives), base from sgpr<0>:
  //       CHECK:   s_add_u32 {{.*}} ins(%[[BASE]], %[[BID]]) : {{.*}} ins(!amdgcn.sgpr<0>, !amdgcn.sgpr<2>)
  amdgcn.kernel @fixed_livein_copy attributes {grid_dims = array<i32: 8, 1, 1>} {
    // fixed live-in without writer, e.g. block_id x implicit mapping
    %bid  = amdgcn.alloca : !amdgcn.sgpr<2>
    %g    = amdgcn.alloca : !amdgcn.sgpr
    %cp   = lsir.copy %g, %bid : !amdgcn.sgpr, !amdgcn.sgpr<2>
    %base = amdgcn.alloca : !amdgcn.sgpr
    %acc  = amdgcn.alloca : !amdgcn.sgpr
    %cc   = amdgcn.alloca : !amdgcn.scc<0>
    %c0   = arith.constant 0 : i32
    %b    = amdgcn.s_mov_b32 outs(%base) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    %r    = amdgcn.s_add_u32 outs(%acc, %cc) ins(%b, %cp) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %r : (!amdgcn.sgpr) -> ()
    amdgcn.end_kernel
  }

  // CHECK-LABEL: kernel @fixed_livein_copy_bbarg
  //       CHECK:   %[[BID:.*]] = alloca : !amdgcn.sgpr<2>
  //   CHECK-NOT:   lsir.copy
  //       CHECK:   lsir.br ^bb1
  //       CHECK: ^bb1:
  //   block_id read from sgpr<2> across the loop carry, not a clobbered reg:
  //       CHECK:   s_add_u32 {{.*}} ins({{.*}}, %[[BID]]) : {{.*}} ins({{.*}}, !amdgcn.sgpr<2>)
  //       CHECK:   s_add_u32 {{.*}} ins({{.*}}) : {{.*}} ins({{.*}}, i32)
  amdgcn.kernel @fixed_livein_copy_bbarg attributes {grid_dims = array<i32: 8, 1, 1>} {
    // fixed live-in without writer, e.g. block_id x implicit mapping
    %bid   = amdgcn.alloca : !amdgcn.sgpr<2>
    %g     = amdgcn.alloca : !amdgcn.sgpr
    %cp    = lsir.copy %g, %bid : !amdgcn.sgpr, !amdgcn.sgpr<2>
    %s0    = amdgcn.alloca : !amdgcn.sgpr
    %s1    = amdgcn.alloca : !amdgcn.sgpr
    %cc0   = amdgcn.alloca : !amdgcn.scc<0>
    %scc   = amdgcn.alloca : !amdgcn.scc
    %c0    = arith.constant 0 : i32
    %c1    = arith.constant 1 : i32
    %c4    = arith.constant 4 : i32
    %base  = amdgcn.s_mov_b32 outs(%s0) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    %iter0 = amdgcn.s_mov_b32 outs(%s1) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    lsir.br ^bb1(%iter0, %cp : !amdgcn.sgpr, !amdgcn.sgpr)
  ^bb1(%iter: !amdgcn.sgpr, %bidc: !amdgcn.sgpr):
    %acc  = amdgcn.s_add_u32 outs(%s0, %cc0) ins(%base, %bidc) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    %ni   = amdgcn.s_add_u32 outs(%s1, %cc0) ins(%iter, %c1)   : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, i32)
    %done = amdgcn.s_cmp_lt_i32 outs(%scc) ins(%ni, %c4)       : outs(!amdgcn.scc) ins(!amdgcn.sgpr, i32)
    lsir.cond_br %done : !amdgcn.scc, ^bb1(%ni, %bidc : !amdgcn.sgpr, !amdgcn.sgpr), ^bb2
  ^bb2:
    amdgcn.end_kernel
  }

  // PRESSURE-LABEL: kernel @pressure_fixed_livein
  //       PRESSURE:   alloca : !amdgcn.sgpr<2>
  //   vd is pushed to sgpr<3>; block_id is read from its fixed sgpr<2>:
  //       PRESSURE:   s_add_u32 {{.*}} ins(!amdgcn.sgpr<0>, !amdgcn.sgpr<1>)
  //       PRESSURE:   s_add_u32 {{.*}} ins(!amdgcn.sgpr<3>, !amdgcn.sgpr<2>)
  amdgcn.kernel @pressure_fixed_livein attributes {grid_dims = array<i32: 8, 1, 1>} {
    %c0   = arith.constant 0 : i32
    // stress-test coalescing with a fixed live-in (block_id x) under pressure
    // (i.e. only 4 SGPR registers available for regalloc).
    %sa   = amdgcn.alloca : !amdgcn.sgpr
    %sb   = amdgcn.alloca : !amdgcn.sgpr
    %sd   = amdgcn.alloca : !amdgcn.sgpr
    %va   = amdgcn.s_mov_b32 outs(%sa) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    %vb   = amdgcn.s_mov_b32 outs(%sb) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    %vd   = amdgcn.s_mov_b32 outs(%sd) ins(%c0) : outs(!amdgcn.sgpr) ins(i32)
    %bid  = amdgcn.alloca : !amdgcn.sgpr<2>     // fixed live-in (no writer), like block_id x
    %sg   = amdgcn.alloca : !amdgcn.sgpr
    %cp   = lsir.copy %sg, %bid : !amdgcn.sgpr, !amdgcn.sgpr<2>
    %sr1  = amdgcn.alloca : !amdgcn.sgpr
    %sr2  = amdgcn.alloca : !amdgcn.sgpr
    %scc0 = amdgcn.alloca : !amdgcn.scc<0>
    %r1   = amdgcn.s_add_u32 outs(%sr1, %scc0) ins(%va, %vb) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    %r2   = amdgcn.s_add_u32 outs(%sr2, %scc0) ins(%vd, %cp) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
    test_inst ins %r1 : (!amdgcn.sgpr) -> ()
    test_inst ins %r2 : (!amdgcn.sgpr) -> ()
    amdgcn.end_kernel
  }
}
