// RUN: aster-opt %s --pass-pipeline=" \
// RUN:  builtin.module( \
// RUN:    aster-to-int-arith,\
// RUN:    aster-optimize-arith, \
// RUN:    func.func( \
// RUN:      aster-amdgcn-set-abi \
// RUN:    ), \
// RUN:    aster-codegen, \
// RUN:    canonicalize, \
// RUN:    aster-to-amdgcn, \
// RUN:    amdgcn.kernel( \
// RUN:      aster-amdgcn-expand-md-ops, \
// RUN:      amdgcn-register-allocation \
// RUN:    ))" | FileCheck %s


// CHECK-LABEL:  amdgcn.kernel @tid
// CHECK-SAME:     arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = !amdgcn.sgpr>, #amdgcn.by_val_arg<size = 4, alignment = 4, type = !amdgcn.sgpr>, #amdgcn.by_val_arg<size = 4, alignment = 4, type = !amdgcn.sgpr>, #amdgcn.block_dim_arg<x>, #amdgcn.block_dim_arg<y>, #amdgcn.block_dim_arg<z>, #amdgcn.grid_dim_arg<x>, #amdgcn.grid_dim_arg<y>, #amdgcn.grid_dim_arg<z>]> {
// CHECK:          %[[C7:.*]] = arith.constant 7 : i32
// CHECK:          sop2 s_lshl_b32 outs %2 ins %1, %[[C7]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<2>, i32
// CHECK:          vop2 v_add_u32 outs %{{.*}} ins %{{.*}}, %{{.*}} : !amdgcn.vgpr<0>, !amdgcn.sgpr<0>, !amdgcn.vgpr<0>
// CHECK:          test_inst ins %{{.*}} : (!amdgcn.vgpr<0>) -> ()
// CHECK:          end_kernel
#map = affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>
func.func @tid(%arg0: index, %arg1: index, %arg2: index) attributes {gpu.block_dims = array<i32: 128, 1, 1>, gpu.grid_dims = array<i32: 512, 1, 1>, gpu.kernel} {
  %thread_id_x = gpu.thread_id  x
  %block_id_x = gpu.block_id  x
  %block_dim_x = gpu.block_dim  x
  %0 = affine.apply #map()[%thread_id_x, %block_id_x, %block_dim_x]
  %1 = arith.index_cast %0 : index to i32
  %2 = lsir.to_reg %1 : i32 -> !amdgcn.vgpr
  amdgcn.test_inst ins %2 : (!amdgcn.vgpr) -> ()
  return
}
