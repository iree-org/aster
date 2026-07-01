// RUN: aster-opt %s --split-input-file --verify-diagnostics

// 2*2*5 = 20 is just over the limit.
amdgcn.module @too_large_20 target = #amdgcn.target<gfx1250> {
  // expected-error @below {{cluster size (product of cluster_dims) must be <= 16, got 20}}
  amdgcn.kernel @k attributes {cluster_dims = array<i32: 2, 2, 5>} {
    amdgcn.end_kernel
  }
}

// -----

// 2*2*4 = 16 is exactly the limit and must verify cleanly (no diagnostic).
amdgcn.module @at_limit_16 target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k attributes {cluster_dims = array<i32: 2, 2, 4>} {
    amdgcn.end_kernel
  }
}

// -----

// Default {0,0,0} (no clustering) verifies cleanly.
amdgcn.module @no_cluster target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k {
    amdgcn.end_kernel
  }
}

// -----

// When grid_dims is known it must tile evenly into clusters: 7 is not a
// multiple of cluster z = 3.
amdgcn.module @grid_not_multiple target = #amdgcn.target<gfx1250> {
  // expected-error @below {{grid_dims[2] (7) must be an exact multiple of cluster_dims[2] (3)}}
  amdgcn.kernel @k attributes {cluster_dims = array<i32: 1, 2, 3>,
                               grid_dims = array<i32: 2, 4, 7>} {
    amdgcn.end_kernel
  }
}

// -----

// grid_dims an exact multiple of cluster_dims per axis verifies cleanly.
amdgcn.module @grid_multiple_ok target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @k attributes {cluster_dims = array<i32: 1, 2, 3>,
                               grid_dims = array<i32: 2, 4, 6>} {
    amdgcn.end_kernel
  }
}
