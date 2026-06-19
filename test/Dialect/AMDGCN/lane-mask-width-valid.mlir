// RUN: aster-opt %s --split-input-file

// LaneMaskWidthTrait accepts correct-width lane masks.
amdgcn.module @wave32_vcc_lo target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %vl = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %c0 = arith.constant 0 : i32
    amdgcn.v_cmp_lt_i32 outs(%vl) ins(%v0, %c0) : outs(!amdgcn.vcc_lo<0>) ins(!amdgcn.vgpr<0>, i32)
    amdgcn.s_cbranch_vccnz %vl, true(^t) false(^f) : !amdgcn.vcc_lo<0>
  ^f:
    amdgcn.end_kernel
  ^t:
    amdgcn.end_kernel
  }
}

// -----

// wave64 (gfx942): the 64-bit vcc is accepted on v_cmp and s_cbranch.
amdgcn.module @wave64_vcc target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %vl = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %vh = amdgcn.alloca : !amdgcn.vcc_hi<0>
    %vcc = amdgcn.make_register_range %vl, %vh : !amdgcn.vcc_lo<0>, !amdgcn.vcc_hi<0>
    %c0 = arith.constant 0 : i32
    amdgcn.v_cmp_lt_i32 outs(%vcc) ins(%v0, %c0) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vgpr<0>, i32)
    amdgcn.s_cbranch_vccnz %vcc, true(^t) false(^f) : !amdgcn.vcc<0>
  ^f:
    amdgcn.end_kernel
  ^t:
    amdgcn.end_kernel
  }
}
