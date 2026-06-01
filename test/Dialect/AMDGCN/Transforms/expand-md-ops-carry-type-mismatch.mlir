// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<
    !ptr.ptr<#amdgcn.addr_space<local, read_write>> = #ptr.spec<size = 32, abi = 32, preferred = 32>,
    !ptr.ptr<#amdgcn.addr_space<global, read_write>> = #ptr.spec<size = 64, abi = 64, preferred = 64>>} {

// CHECK-LABEL: kernel @block_id_carry_type_mismatch
//       CHECK:     %[[BID:.*]] = alloca : !amdgcn.sgpr<2>
//       CHECK:     %[[CP:.*]] = lsir.copy %{{.*}}, %[[BID]] : !amdgcn.sgpr, !amdgcn.sgpr<2>
//       CHECK:     lsir.br ^bb1(%{{.*}}, %[[CP]] : !amdgcn.sgpr, !amdgcn.sgpr)
//       CHECK:   ^bb1(%{{.*}}: !amdgcn.sgpr, %[[CARRY:.*]]: !amdgcn.sgpr):
//       CHECK:     s_add_u32 {{.*}} ins({{.*}}, %[[CARRY]]) : {{.*}} ins(!amdgcn.sgpr, !amdgcn.sgpr)
  amdgcn.module @bug61 target = <gfx942>
      attributes {normal_forms = [#amdgcn.no_reg_cast_ops, #amdgcn.no_lsir_compute_ops]} {
    kernel @block_id_carry_type_mismatch arguments <>
        attributes {grid_dims = array<i32: 8, 1, 1>} {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %s0  = alloca : !amdgcn.sgpr
      %s1  = alloca : !amdgcn.sgpr
      %cc0 = alloca : !amdgcn.scc<0>
      %scc = alloca : !amdgcn.scc
      %bid_in = block_id x : !amdgcn.sgpr
      %base  = s_mov_b32 outs(%s0) ins(%c0_i32) : outs(!amdgcn.sgpr) ins(i32)
      %iter0 = s_mov_b32 outs(%s1) ins(%c0_i32) : outs(!amdgcn.sgpr) ins(i32)
      lsir.br ^bb1(%iter0, %bid_in : !amdgcn.sgpr, !amdgcn.sgpr)
    ^bb1(%iter: !amdgcn.sgpr, %bid: !amdgcn.sgpr):
      %acc  = s_add_u32 outs(%s0, %cc0) ins(%base, %bid)    : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
      %ni   = s_add_u32 outs(%s1, %cc0) ins(%iter, %c1_i32) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, i32)
      %done = s_cmp_lt_i32 outs(%scc)   ins(%ni, %c4_i32)   : outs(!amdgcn.scc) ins(!amdgcn.sgpr, i32)
      lsir.cond_br %done : !amdgcn.scc,
        ^bb1(%ni, %bid : !amdgcn.sgpr, !amdgcn.sgpr),
        ^bb2
    ^bb2:
      end_kernel
    }
  }
}
