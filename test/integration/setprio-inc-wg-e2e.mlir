// RUN: aster-opt %s --verify-roundtrip

amdgcn.module @setprio_inc_wg_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @setprio_inc_wg {
    s_setprio_inc_wg 100
    s_setprio_inc_wg 0
    amdgcn.end_kernel
  }
}
