// RUN: aster-opt %s --split-input-file --verify-diagnostics

// LaneMaskWidthTrait: lane-mask width must match module wave size.
amdgcn.module @cbranch_vcc64_on_wave32 target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k {
  ^entry:
    %l = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %h = amdgcn.alloca : !amdgcn.vcc_hi<0>
    %vcc = amdgcn.make_register_range %l, %h : !amdgcn.vcc_lo<0>, !amdgcn.vcc_hi<0>
    // expected-error @+1 {{VCC operand must match the wave size}}
    amdgcn.s_cbranch_vccnz %vcc, true(^t) false(^f) : !amdgcn.vcc<0>
  ^f:
    amdgcn.end_kernel
  ^t:
    amdgcn.end_kernel
  }
}

// -----

// wave64: s_cbranch rejects VCC_LO.
amdgcn.module @cbranch_vcc_lo_on_wave64 target = #amdgcn.target<gfx942> {
  amdgcn.kernel @k {
  ^entry:
    %l = amdgcn.alloca : !amdgcn.vcc_lo<0>
    // expected-error @+1 {{VCC operand must match the wave size}}
    amdgcn.s_cbranch_vccz %l, true(^t) false(^f) : !amdgcn.vcc_lo<0>
  ^f:
    amdgcn.end_kernel
  ^t:
    amdgcn.end_kernel
  }
}

// -----

// wave32: v_cmp rejects 64-bit VCC.
amdgcn.module @vcmp_vcc64_on_wave32 target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %l = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %h = amdgcn.alloca : !amdgcn.vcc_hi<0>
    %vcc = amdgcn.make_register_range %l, %h : !amdgcn.vcc_lo<0>, !amdgcn.vcc_hi<0>
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{VCC operand must match the wave size}}
    amdgcn.v_cmp_lt_i32 outs(%vcc) ins(%v0, %c0) : outs(!amdgcn.vcc<0>) ins(!amdgcn.vgpr<0>, i32)
    amdgcn.end_kernel
  }
}

// -----

// wave32: v_cndmask rejects 64-bit VCC.
amdgcn.module @vcndmask_vcc64_on_wave32 target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k {
  ^entry:
    %v0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %v1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %v2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %l = amdgcn.alloca : !amdgcn.vcc_lo<0>
    %h = amdgcn.alloca : !amdgcn.vcc_hi<0>
    %vcc = amdgcn.make_register_range %l, %h : !amdgcn.vcc_lo<0>, !amdgcn.vcc_hi<0>
    // expected-error @+1 {{VCC operand must match the wave size}}
    amdgcn.v_cndmask_b32 outs(%v2) ins(%v0, %v1, %vcc) : outs(!amdgcn.vgpr<2>) ins(!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vcc<0>)
    amdgcn.end_kernel
  }
}
