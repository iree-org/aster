// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// .cluster_dims kernel metadata is emitted when the cluster_dims attribute is
// set, and omitted when it defaults to {0,0,0}.

// CHECK-LABEL: amdhsa.kernels:
//      CHECK: .cluster_dims: [ 2, 2, 1 ]
//      CHECK: .name: with_clusters
//  CHECK-NOT: .cluster_dims
//      CHECK: .name: without_clusters

amdgcn.module @cluster_dims_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @with_clusters attributes {cluster_dims = array<i32: 2, 2, 1>} {
    amdgcn.end_kernel
  }
  amdgcn.kernel @without_clusters {
    amdgcn.end_kernel
  }
}
