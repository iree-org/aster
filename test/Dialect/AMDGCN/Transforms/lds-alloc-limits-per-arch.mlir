// RUN: aster-opt --amdgcn-lds-alloc --split-input-file --verify-diagnostics %s

// gfx1250 has 320 KB LDS.
amdgcn.module @gfx1250_fits target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k {
    %0 = amdgcn.alloc_lds 300000
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @gfx1251_fits target = #amdgcn.target<gfx1251> {
  amdgcn.kernel @k {
    %0 = amdgcn.alloc_lds 300000
    amdgcn.end_kernel
  }
}

// -----

amdgcn.module @gfx1250_overflow target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k {
    // expected-error@+1 {{failed to allocate LDS buffer of size 400000 with alignment 16; would exceed 327680 bytes}}
    %0 = amdgcn.alloc_lds 400000
    amdgcn.end_kernel
  }
}

// -----

// gfx940 has 64 KB LDS.
amdgcn.module @cdna3_gfx940_overflow target = #amdgcn.target<gfx940> {
  amdgcn.kernel @k {
    // expected-error@+1 {{failed to allocate LDS buffer of size 300000 with alignment 16; would exceed 65536 bytes}}
    %0 = amdgcn.alloc_lds 300000
    amdgcn.end_kernel
  }
}
