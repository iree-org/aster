// RUN: aster-opt %s --verify-diagnostics --split-input-file

// gfx1250 s_wait_*cnt counts are confined to their HW field max: 6-bit (max 63)
// for load/store/ds/tensor/async, 5-bit (max 31) for km.

amdgcn.module @m target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @loadcnt_over_max attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    // expected-error@+1 {{attribute 'count' failed to satisfy constraint: LOAD_CNT wait count}}
    amdgcn.s_wait_loadcnt 64
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @m target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @asynccnt_over_max attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    // expected-error@+1 {{attribute 'count' failed to satisfy constraint: ASYNC_CNT wait count}}
    amdgcn.s_wait_asynccnt 64
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @m target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @kmcnt_over_max attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    // expected-error@+1 {{attribute 'count' failed to satisfy constraint: KM_CNT wait count}}
    amdgcn.s_wait_kmcnt 32
    amdgcn.end_kernel
  }
}
