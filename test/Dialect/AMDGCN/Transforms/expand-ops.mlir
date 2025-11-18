// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops, aster-hoist-ops)))" | FileCheck %s

amdgcn.module @kernel_with_ptr target = <gfx940> isa = <cdna3> {
// CHECK-LABEL:   @kernel_ptr
// CHECK:        %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:         %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:         %[[VAL_2:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_3:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_4:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_5:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_6:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_7:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_8:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:         %[[VAL_9:.*]] = make_register_range %[[VAL_2]], %[[VAL_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[SMEM_0:.*]] = amdgcn.smem.load <s_load_dwordx2> %[[VAL_9]], %[[VAL_8]] : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr_range<[? + 2]>
// CHECK:         %[[VAL_10:.*]] = make_register_range %[[VAL_4]], %[[VAL_5]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[SMEM_1:.*]] = amdgcn.smem.load <s_load_dwordx2> %[[VAL_10]], %[[VAL_8]] offset = 8 : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr_range<[? + 2]>
// CHECK:         %[[VAL_11:.*]] = make_register_range %[[VAL_6]], %[[VAL_7]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[SMEM_2:.*]] = amdgcn.smem.load <s_load_dwordx2> %[[VAL_11]], %[[VAL_8]] offset = 16 : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr_range<[? + 2]>
// CHECK:         test_inst ins %[[SMEM_0]], %[[SMEM_1]], %[[SMEM_2]] : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) -> ()
// CHECK:         end_kernel
  kernel @kernel_ptr arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = write_only>,
      #amdgcn.buffer_arg<address_space = private, access = read_only, flags = const|volatile>,
      #amdgcn.buffer_arg<address_space = generic, type = !ptr.ptr<#ptr.generic_space>>
    ]> {
    %0 = load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %1 = load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    %2 = load_arg 2 : !amdgcn.sgpr_range<[? + 2]>
    test_inst ins %0, %1, %2 : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) -> ()
    end_kernel
  }

// CHECK-LABEL:   @byval
// CHECK:         %[[VAL_12:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:         %[[VAL_13:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:         %[[VAL_14:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_15:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_16:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_17:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_18:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_19:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_20:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_21:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_22:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_23:.*]] = make_register_range %[[VAL_12]], %[[VAL_13]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:         %[[VAL_24:.*]] = make_register_range %[[VAL_14]], %[[VAL_15]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[VAL_25:.*]]:2 = split_register_range %[[VAL_24]] : !amdgcn.sgpr_range<[? + 2]>
// CHECK:         %[[SMEM_3:.*]] = amdgcn.smem.load <s_load_dword> %[[VAL_25]]#0, %[[VAL_23]] : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:         %[[SMEM_4:.*]] = amdgcn.smem.load <s_load_dword> %[[VAL_25]]#1, %[[VAL_23]] offset = 4 : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:         %[[VAL_26:.*]] = make_register_range %[[SMEM_3]], %[[SMEM_4]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[SMEM_5:.*]] = amdgcn.smem.load <s_load_dword> %[[VAL_16]], %[[VAL_23]] offset = 8 : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:         %[[VAL_27:.*]] = make_register_range %[[VAL_17]], %[[VAL_18]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[SMEM_6:.*]] = amdgcn.smem.load <s_load_dwordx2> %[[VAL_27]], %[[VAL_23]] offset = 16 : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr_range<[? + 2]>
// CHECK:         %[[VAL_28:.*]] = make_register_range %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[SMEM_7:.*]] = amdgcn.smem.load <s_load_dwordx4> %[[VAL_28]], %[[VAL_23]] offset = 24 : !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr_range<[? + 4]>
// CHECK:         test_inst ins %[[VAL_26]], %[[SMEM_5]], %[[SMEM_6]], %[[SMEM_7]] : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 4]>) -> ()
// CHECK:         end_kernel
  kernel @byval arguments <[
      #amdgcn.by_val_arg<size = 6, alignment = 8, type = i48>,
      #amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>,
      #amdgcn.by_val_arg<size = 8, alignment = 8, type = i64>,
      #amdgcn.by_val_arg<size = 16, alignment = 8, type = i128>
    ]> {
    %0 = load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %1 = load_arg 1 : !amdgcn.sgpr
    %2 = load_arg 2 : !amdgcn.sgpr_range<[? + 2]>
    %3 = load_arg 3 : !amdgcn.sgpr_range<[? + 4]>
    test_inst ins %0, %1, %2, %3 : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 4]>) -> ()
    end_kernel
  }
// CHECK-LABEL:   @thread_block_x
// CHECK:             %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:             %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK:             test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<2>) -> ()
  kernel @thread_block_x arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  x : !amdgcn.vgpr
    %1 = block_id  x : !amdgcn.sgpr
    test_inst ins %0, %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }
// CHECK-LABEL:   @thread_block_ids
// CHECK:         attributes {enable_workgroup_id_y, enable_workgroup_id_z, workitem_id_mode = #amdgcn.workitem_id_mode<x_y_z>}
// CHECK:         %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:         %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:         %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK:         %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK:         %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK:         %[[VAL_5:.*]] = alloca : !amdgcn.sgpr<4>
// CHECK:         test_inst ins %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>, !amdgcn.sgpr<4>) -> ()
// CHECK:         end_kernel
  kernel @thread_block_ids arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  x : !amdgcn.vgpr
    %1 = thread_id  y : !amdgcn.vgpr
    %2 = thread_id  z : !amdgcn.vgpr
    %3 = block_id  x : !amdgcn.sgpr
    %4 = block_id  y : !amdgcn.sgpr
    %5 = block_id  z : !amdgcn.sgpr
    test_inst ins %0, %1, %2, %3, %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }

// CHECK-LABEL:  @grid_block_dim arguments
// CHECK-SAME:       <[#amdgcn.block_dim_arg<x>, #amdgcn.block_dim_arg<y>, #amdgcn.block_dim_arg<z>, #amdgcn.grid_dim_arg<x>, #amdgcn.grid_dim_arg<y>, #amdgcn.grid_dim_arg<z>]> attributes {enable_workgroup_id_x = false} {
// CHECK:            %[[S0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK:            %[[S1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK:            %[[S01:.*]] = make_register_range %[[S0]], %[[S1]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:            %{{.*}} = amdgcn.smem.load <s_load_dword> %{{.*}}, %[[S01]] offset = 12 : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:            %{{.*}} = amdgcn.smem.load <s_load_dword> %{{.*}}, %[[S01]] offset = 16 : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:            %{{.*}} = amdgcn.smem.load <s_load_dword> %{{.*}}, %[[S01]] offset = 20 : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:            %{{.*}} = amdgcn.smem.load <s_load_dword> %{{.*}}, %[[S01]] : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:            %{{.*}} = amdgcn.smem.load <s_load_dword> %{{.*}}, %[[S01]] offset = 4 : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:            %{{.*}} = amdgcn.smem.load <s_load_dword> %{{.*}}, %[[S01]] offset = 8 : !amdgcn.sgpr, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.sgpr
// CHECK:            end_kernel
// CHECK:          }
  kernel @grid_block_dim {
    %0 = grid_dim  x : !amdgcn.sgpr
    %1 = grid_dim  y : !amdgcn.sgpr
    %2 = grid_dim  z : !amdgcn.sgpr
    %3 = block_dim  x : !amdgcn.sgpr
    %4 = block_dim  y : !amdgcn.sgpr
    %5 = block_dim  z : !amdgcn.sgpr
    test_inst ins %0, %1, %2, %3, %4, %5 : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }
}
