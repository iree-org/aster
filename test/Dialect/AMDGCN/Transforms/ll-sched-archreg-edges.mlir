// RUN: aster-opt %s --pass-pipeline="builtin.module(test-amdgcn-sched-graph)" 2>&1 | FileCheck %s

!v   = !amdgcn.vgpr
!s   = !amdgcn.sgpr

amdgcn.module @scc_war target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @scc_war
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "s_addc_u32 -> s_add_u32"
  // CHECK-NOT:     label = "s_add_u32 -> s_addc_u32"
  // CHECK:       }
  amdgcn.kernel @scc_war {
    %s_a    = amdgcn.alloca : !s
    %s_b    = amdgcn.alloca : !s
    %s_d1   = amdgcn.alloca : !s
    %s_d2   = amdgcn.alloca : !s
    %scc_in  = amdgcn.alloca : !amdgcn.scc<0>
    %scc_out = amdgcn.alloca : !amdgcn.scc<0>
    %addc = amdgcn.s_addc_u32 outs(%s_d1, %scc_out) ins(%s_a, %s_b, %scc_in) : outs(!s, !amdgcn.scc<0>) ins(!s, !s, !amdgcn.scc<0>)
    %add  = amdgcn.s_add_u32  outs(%s_d2, %scc_in)  ins(%s_a, %s_b)          : outs(!s, !amdgcn.scc<0>) ins(!s, !s)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @vcc_raw target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @vcc_raw
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "v_cmp_eq_i32 -> v_cndmask_b32"
  // CHECK-NOT:     label = "v_cndmask_b32 -> v_cmp_eq_i32"
  // CHECK:       }
  amdgcn.kernel @vcc_raw {
    %v_a = amdgcn.alloca : !v
    %v_b = amdgcn.alloca : !v
    %v_d = amdgcn.alloca : !v
    %vcc = amdgcn.alloca : !amdgcn.vcc<0>
    amdgcn.v_cmp_eq_i32 outs(%vcc) ins(%v_a, %v_b) : outs(!amdgcn.vcc<0>) ins(!v, !v)
    %sel = amdgcn.v_cndmask_b32 outs(%v_d) ins(%v_a, %v_b, %vcc) : outs(!v) ins(!v, !v, !amdgcn.vcc<0>)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @scc_waw_relaxed target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @scc_waw_relaxed
  // CHECK:       digraph SchedGraph
  // CHECK-NOT:     label = "s_add_u32 -> s_and_b32"
  // CHECK-NOT:     label = "s_and_b32 -> s_add_u32"
  // CHECK:       }
  amdgcn.kernel @scc_waw_relaxed {
    %s_a   = amdgcn.alloca : !s
    %s_b   = amdgcn.alloca : !s
    %s_d1  = amdgcn.alloca : !s
    %s_d2  = amdgcn.alloca : !s
    %scc_a = amdgcn.alloca : !amdgcn.scc<0>
    %scc_b = amdgcn.alloca : !amdgcn.scc<0>
    %add = amdgcn.s_add_u32 outs(%s_d1, %scc_a) ins(%s_a, %s_b) : outs(!s, !amdgcn.scc<0>) ins(!s, !s)
    %and = amdgcn.s_and_b32 outs(%s_d2, %scc_b) ins(%s_a, %s_b) : outs(!s, !amdgcn.scc<0>) ins(!s, !s)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @scc_war_through_writer_run target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @scc_war_through_writer_run
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "s_addc_u32 -> s_add_u32"
  // CHECK-DAG:     label = "s_addc_u32 -> s_and_b32"
  // CHECK:       }
  amdgcn.kernel @scc_war_through_writer_run {
    %s_a    = amdgcn.alloca : !s
    %s_b    = amdgcn.alloca : !s
    %s_d1   = amdgcn.alloca : !s
    %s_d2   = amdgcn.alloca : !s
    %s_d3   = amdgcn.alloca : !s
    %scc_in  = amdgcn.alloca : !amdgcn.scc<0>
    %scc_a   = amdgcn.alloca : !amdgcn.scc<0>
    %scc_b   = amdgcn.alloca : !amdgcn.scc<0>
    %addc = amdgcn.s_addc_u32 outs(%s_d1, %scc_a) ins(%s_a, %s_b, %scc_in) : outs(!s, !amdgcn.scc<0>) ins(!s, !s, !amdgcn.scc<0>)
    %add  = amdgcn.s_add_u32  outs(%s_d2, %scc_in) ins(%s_a, %s_b) : outs(!s, !amdgcn.scc<0>) ins(!s, !s)
    %and  = amdgcn.s_and_b32  outs(%s_d3, %scc_b) ins(%s_a, %s_b) : outs(!s, !amdgcn.scc<0>) ins(!s, !s)
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @scc_vcc_independent target = #amdgcn.target<gfx942> {
  // CHECK-LABEL: Kernel: @scc_vcc_independent
  // CHECK:       digraph SchedGraph
  // CHECK-DAG:     label = "s_add_u32 -> s_addc_u32"
  // CHECK-DAG:     label = "v_cmp_eq_i32 -> v_cndmask_b32"
  // CHECK-NOT:     label = "s_add_u32 -> v_cmp_eq_i32"
  // CHECK-NOT:     label = "v_cmp_eq_i32 -> s_add_u32"
  // CHECK-NOT:     label = "s_add_u32 -> v_cndmask_b32"
  // CHECK-NOT:     label = "v_cndmask_b32 -> s_add_u32"
  // CHECK-NOT:     label = "s_addc_u32 -> v_cmp_eq_i32"
  // CHECK-NOT:     label = "v_cmp_eq_i32 -> s_addc_u32"
  // CHECK-NOT:     label = "s_addc_u32 -> v_cndmask_b32"
  // CHECK:       }
  amdgcn.kernel @scc_vcc_independent {
    %v_a = amdgcn.alloca : !v
    %v_b = amdgcn.alloca : !v
    %v_d = amdgcn.alloca : !v
    %s_a = amdgcn.alloca : !s
    %s_b = amdgcn.alloca : !s
    %s_d1 = amdgcn.alloca : !s
    %s_d2 = amdgcn.alloca : !s
    %vcc   = amdgcn.alloca : !amdgcn.vcc<0>
    %scc_w = amdgcn.alloca : !amdgcn.scc<0>
    %scc_r = amdgcn.alloca : !amdgcn.scc<0>
    amdgcn.v_cmp_eq_i32 outs(%vcc) ins(%v_a, %v_b) : outs(!amdgcn.vcc<0>) ins(!v, !v)
    %add  = amdgcn.s_add_u32  outs(%s_d1, %scc_w) ins(%s_a, %s_b) : outs(!s, !amdgcn.scc<0>) ins(!s, !s)
    %sel  = amdgcn.v_cndmask_b32 outs(%v_d) ins(%v_a, %v_b, %vcc) : outs(!v) ins(!v, !v, !amdgcn.vcc<0>)
    %addc = amdgcn.s_addc_u32 outs(%s_d2, %scc_r) ins(%s_a, %s_b, %scc_w) : outs(!s, !amdgcn.scc<0>) ins(!s, !s, !amdgcn.scc<0>)
    amdgcn.end_kernel
  }
}
