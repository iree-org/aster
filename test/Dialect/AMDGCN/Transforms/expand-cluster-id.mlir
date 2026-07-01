// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" | FileCheck %s

// cluster_id x/y/z -> ttmp9 full (ctrl 0x200000), ttmp7 [15:0] (0x100000),
// ttmp7 [31:16] (0x100010).
// CHECK-LABEL: kernel @cluster_id
//       CHECK:   arith.constant 2097152 : i32
//       CHECK:   s_bfe_u32{{.*}}ins(!amdgcn.ttmp<9>, i32)
//       CHECK:   arith.constant 1048576 : i32
//       CHECK:   s_bfe_u32{{.*}}ins(!amdgcn.ttmp<7>, i32)
//       CHECK:   arith.constant 1048592 : i32
//       CHECK:   s_bfe_u32{{.*}}ins(!amdgcn.ttmp<7>, i32)
amdgcn.module @m_cid target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @cluster_id {
    %x = amdgcn.cluster_id x : !amdgcn.sgpr
    %y = amdgcn.cluster_id y : !amdgcn.sgpr
    %z = amdgcn.cluster_id z : !amdgcn.sgpr
    amdgcn.end_kernel
  }
}

// cluster_workgroup_id x -> ttmp6 [3:0] (ctrl 0x40000).
// CHECK-LABEL: kernel @cluster_wg_id
//       CHECK:   arith.constant 262144 : i32
//       CHECK:   s_bfe_u32{{.*}}ins(!amdgcn.ttmp<6>, i32)
amdgcn.module @m_wgid target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @cluster_wg_id {
    %x = amdgcn.cluster_workgroup_id x : !amdgcn.sgpr
    amdgcn.end_kernel
  }
}

// cluster_workgroup_max_id x -> ttmp6 [15:12] (ctrl 0x4000c).
// CHECK-LABEL: kernel @cluster_wg_max_id
//       CHECK:   arith.constant 262156 : i32
//       CHECK:   s_bfe_u32{{.*}}ins(!amdgcn.ttmp<6>, i32)
amdgcn.module @m_maxid target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @cluster_wg_max_id {
    %x = amdgcn.cluster_workgroup_max_id x : !amdgcn.sgpr
    amdgcn.end_kernel
  }
}

// On GFX12.5 block_id x with clusters is reconstructed as
// TTMP9 (grid/cluster position) * cluster_size + TTMP6.wg_x (ttmp6[3:0]).
// CHECK-LABEL: kernel @block_id_cluster
//       CHECK:   arith.constant 2097152 : i32
//       CHECK:   s_bfe_u32{{.*}}ins(!amdgcn.ttmp<9>, i32)
//       CHECK:   s_mul_i32
//       CHECK:   arith.constant 262144 : i32
//       CHECK:   s_bfe_u32{{.*}}ins(!amdgcn.ttmp<6>, i32)
//       CHECK:   s_add_u32
amdgcn.module @m_bid_c target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @block_id_cluster attributes {cluster_dims = array<i32: 4, 1, 1>} {
    %x = amdgcn.block_id x : !amdgcn.sgpr
    amdgcn.end_kernel
  }
}

// On GFX12.5 without clusters block_id x is just TTMP9 (grid position == wg id);
// no scale/offset is emitted.
// CHECK-LABEL: kernel @block_id_no_cluster
//       CHECK:   arith.constant 2097152 : i32
//       CHECK:   s_bfe_u32{{.*}}ins(!amdgcn.ttmp<9>, i32)
//   CHECK-NOT:   s_mul_i32
//   CHECK-NOT:   s_add_u32
amdgcn.module @m_bid target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @block_id_no_cluster {
    %x = amdgcn.block_id x : !amdgcn.sgpr
    amdgcn.end_kernel
  }
}
