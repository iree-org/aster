// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops, aster-hoist-ops, canonicalize)))" | FileCheck %s

amdgcn.module @kernel_with_ptr target = <gfx940> isa = <cdna3> {
// CHECK-LABEL: kernel @kernel_ptr arguments <[#amdgcn.buffer_arg<address_space = generic, access = write_only>, #amdgcn.buffer_arg<address_space = private, access = read_only, flags = const|volatile>, #amdgcn.buffer_arg<address_space = generic, type = !ptr.ptr<#ptr.generic_space>>]> attributes {enable_workgroup_id_x = false} {
// CHECK-DAG:     %[[CONSTANT_0:.*]] = arith.constant 16 : i32
// CHECK-DAG:     %[[CONSTANT_1:.*]] = arith.constant 8 : i32
// CHECK-DAG:     %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:     %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:     %[[VAL_2:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_3:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_4:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_5:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_6:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_7:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_8:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:         %[[VAL_9:.*]] = make_register_range %[[VAL_2]], %[[VAL_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[VAL_10:.*]], %[[VAL_11:.*]] = load s_load_dwordx2 dest %[[VAL_9]] addr %[[VAL_8]] offset c(%[[CONSTANT_2]]) : dps(!amdgcn.sgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_12:.*]] = make_register_range %[[VAL_4]], %[[VAL_5]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[VAL_13:.*]], %[[VAL_14:.*]] = load s_load_dwordx2 dest %[[VAL_12]] addr %[[VAL_8]] offset c(%[[CONSTANT_1]]) : dps(!amdgcn.sgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_15:.*]] = make_register_range %[[VAL_6]], %[[VAL_7]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[VAL_16:.*]], %[[VAL_17:.*]] = load s_load_dwordx2 dest %[[VAL_15]] addr %[[VAL_8]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         test_inst ins %[[VAL_10]], %[[VAL_13]], %[[VAL_16]] : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>) -> ()
// CHECK:         end_kernel
// CHECK:       }
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

// CHECK-LABEL: kernel @byval arguments <[#amdgcn.by_val_arg<size = 6, alignment = 8, type = i48>, #amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>, #amdgcn.by_val_arg<size = 8, alignment = 8, type = i64>, #amdgcn.by_val_arg<size = 16, alignment = 8, type = i128>]> attributes {enable_workgroup_id_x = false} {
// CHECK-DAG:     %[[CONSTANT_0:.*]] = arith.constant 24 : i32
// CHECK-DAG:     %[[CONSTANT_1:.*]] = arith.constant 16 : i32
// CHECK-DAG:     %[[CONSTANT_2:.*]] = arith.constant 8 : i32
// CHECK-DAG:     %[[CONSTANT_3:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:     %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:     %[[VAL_2:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_3:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_4:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_5:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_6:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_7:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_8:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_9:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_10:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_11:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:         %[[VAL_12:.*]] = make_register_range %[[VAL_2]], %[[VAL_3]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[VAL_13:.*]]:2 = split_register_range %[[VAL_12]] : !amdgcn.sgpr_range<[? + 2]>
// CHECK:         %[[VAL_14:.*]], %[[VAL_15:.*]] = load s_load_dword dest %[[VAL_13]]#0 addr %[[VAL_11]] offset c(%[[CONSTANT_4]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_16:.*]], %[[VAL_17:.*]] = load s_load_dword dest %[[VAL_13]]#1 addr %[[VAL_11]] offset c(%[[CONSTANT_3]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_18:.*]] = make_register_range %[[VAL_14]], %[[VAL_16]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[VAL_19:.*]], %[[VAL_20:.*]] = load s_load_dword dest %[[VAL_4]] addr %[[VAL_11]] offset c(%[[CONSTANT_2]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_21:.*]] = make_register_range %[[VAL_5]], %[[VAL_6]] : !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[VAL_22:.*]], %[[VAL_23:.*]] = load s_load_dwordx2 dest %[[VAL_21]] addr %[[VAL_11]] offset c(%[[CONSTANT_1]]) : dps(!amdgcn.sgpr_range<[? + 2]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_24:.*]] = make_register_range %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
// CHECK:         %[[VAL_25:.*]], %[[VAL_26:.*]] = load s_load_dwordx4 dest %[[VAL_24]] addr %[[VAL_11]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr_range<[? + 4]>) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         test_inst ins %[[VAL_18]], %[[VAL_19]], %[[VAL_22]], %[[VAL_25]] : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 4]>) -> ()
// CHECK:         end_kernel
// CHECK:       }
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

// CHECK-LABEL: kernel @thread_block_x arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
// CHECK-DAG:     %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK:         test_inst ins %[[VAL_0]], %[[VAL_1]] : (!amdgcn.vgpr<0>, !amdgcn.sgpr<2>) -> ()
// CHECK:         end_kernel
// CHECK:       }
  kernel @thread_block_x arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  x : !amdgcn.vgpr
    %1 = block_id  x : !amdgcn.sgpr
    test_inst ins %0, %1 : (!amdgcn.vgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }

// CHECK-LABEL: kernel @thread_block_ids arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> attributes {enable_workgroup_id_y, enable_workgroup_id_z, workitem_id_mode = #amdgcn.workitem_id_mode<x_y_z>} {
// CHECK-DAG:     %[[VAL_0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[VAL_1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:     %[[VAL_2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:     %[[VAL_3:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:     %[[VAL_4:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK-DAG:     %[[VAL_5:.*]] = alloca : !amdgcn.sgpr<4>
// CHECK:         test_inst ins %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>, !amdgcn.sgpr<4>) -> ()
// CHECK:         end_kernel
// CHECK:       }
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

// CHECK-LABEL: kernel @grid_block_dim arguments <[#amdgcn.block_dim_arg<x>, #amdgcn.block_dim_arg<y>, #amdgcn.block_dim_arg<z>, #amdgcn.grid_dim_arg<x>, #amdgcn.grid_dim_arg<y>, #amdgcn.grid_dim_arg<z>]> attributes {enable_workgroup_id_x = false} {
// CHECK-DAG:     %[[CONSTANT_0:.*]] = arith.constant 8 : i32
// CHECK-DAG:     %[[CONSTANT_1:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[CONSTANT_2:.*]] = arith.constant 65535 : i32
// CHECK-DAG:     %[[CONSTANT_3:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[CONSTANT_4:.*]] = arith.constant 20 : i32
// CHECK-DAG:     %[[CONSTANT_5:.*]] = arith.constant 16 : i32
// CHECK-DAG:     %[[CONSTANT_6:.*]] = arith.constant 12 : i32
// CHECK-DAG:     %[[VAL_0:.*]] = alloca : !amdgcn.sgpr<0>
// CHECK-DAG:     %[[VAL_1:.*]] = alloca : !amdgcn.sgpr<1>
// CHECK-DAG:     %[[VAL_2:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_3:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_4:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_5:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_6:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_7:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_8:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_9:.*]] = alloca : !amdgcn.sgpr
// CHECK-DAG:     %[[VAL_10:.*]] = alloca : !amdgcn.sgpr
// CHECK:         %[[VAL_11:.*]] = make_register_range %[[VAL_0]], %[[VAL_1]] : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
// CHECK:         %[[VAL_12:.*]], %[[VAL_13:.*]] = load s_load_dword dest %[[VAL_2]] addr %[[VAL_11]] offset c(%[[CONSTANT_6]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_14:.*]], %[[VAL_15:.*]] = load s_load_dword dest %[[VAL_3]] addr %[[VAL_11]] offset c(%[[CONSTANT_5]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_16:.*]], %[[VAL_17:.*]] = load s_load_dword dest %[[VAL_4]] addr %[[VAL_11]] offset c(%[[CONSTANT_4]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         %[[VAL_18:.*]], %[[VAL_19:.*]] = load s_load_dword dest %[[VAL_5]] addr %[[VAL_11]] offset c(%[[CONSTANT_3]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
// CHECK:         %[[VAL_20:.*]] = sop2 s_and_b32 outs %[[VAL_6]] ins %[[VAL_18]], %[[CONSTANT_2]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:         %[[VAL_21:.*]], %[[VAL_22:.*]] = load s_load_dword dest %[[VAL_7]] addr %[[VAL_11]] offset c(%[[CONSTANT_1]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
// CHECK:         %[[VAL_23:.*]] = sop2 s_and_b32 outs %[[VAL_8]] ins %[[VAL_21]], %[[CONSTANT_2]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:         %[[VAL_24:.*]], %[[VAL_25:.*]] = load s_load_dword dest %[[VAL_9]] addr %[[VAL_11]] offset c(%[[CONSTANT_0]]) : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[0 : 2]>, i32) -> !amdgcn.read_token<constant>
// CHECK:         amdgcn.sopp.s_waitcnt <s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
// CHECK:         %[[VAL_26:.*]] = sop2 s_and_b32 outs %[[VAL_10]] ins %[[VAL_24]], %[[CONSTANT_2]] : !amdgcn.sgpr, !amdgcn.sgpr, i32
// CHECK:         test_inst ins %[[VAL_12]], %[[VAL_14]], %[[VAL_16]], %[[VAL_20]], %[[VAL_23]], %[[VAL_26]] : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> ()
// CHECK:         end_kernel
// CHECK:       }
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

// CHECK-LABEL: kernel @make_buffer_rsrc_raw
//       CHECK:   %[[BASE:.*]], %{{.*}} = load s_load_dwordx2
//       CHECK:   %[[NREC:.*]], %{{.*}} = load s_load_dword
//       CHECK:   %[[BASE_SPLIT:.*]]:2 = split_register_range %[[BASE]] : !amdgcn.sgpr_range<[? + 2]>
//   CHECK-NOT:   sop2 s_or_b32
//       CHECK:   %[[FLAGS_VAL:.*]] = sop1 s_mov_b32 outs %{{.*}} ins %c0_i32 : !amdgcn.sgpr, i32
//       CHECK:   %[[RSRC:.*]] = make_register_range %[[BASE_SPLIT]]#0, %[[BASE_SPLIT]]#1, %[[NREC]], %[[FLAGS_VAL]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
//       CHECK:   test_inst ins %[[RSRC]] : (!amdgcn.sgpr_range<[? + 4]>) -> ()
//       CHECK:   end_kernel
  kernel @make_buffer_rsrc_raw arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = read_only>,
      #amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>
    ]> {
    %base = load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %num_records = load_arg 1 : !amdgcn.sgpr
    // stride=0, no swizzle -> dword 1 is just base_hi, no s_or needed
    %stride = arith.constant 0 : i32
    %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %stride,
      cache_swizzle = false, swizzle_enable = false, flags = 0
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
    test_inst ins %rsrc : (!amdgcn.sgpr_range<[? + 4]>) -> ()
    end_kernel
  }

// stride=16 -> dword1 upper bits = 16 << 16 = 0x00100000 = 1048576
// CHECK-LABEL: kernel @make_buffer_rsrc_structured
//       CHECK:   %[[BASE2:.*]], %{{.*}} = load s_load_dwordx2
//       CHECK:   %[[NREC2:.*]], %{{.*}} = load s_load_dword
//       CHECK:   %[[BASE_SPLIT2:.*]]:2 = split_register_range %[[BASE2]] : !amdgcn.sgpr_range<[? + 2]>
//       CHECK:   %[[DWORD1:.*]] = sop2 s_or_b32 outs %{{.*}} ins %[[BASE_SPLIT2]]#1, %c1048576_i32 : !amdgcn.sgpr, !amdgcn.sgpr, i32
//       CHECK:   %[[DWORD0_2:.*]] = sop1 s_mov_b32 outs %{{.*}} ins %[[BASE_SPLIT2]]#0 : !amdgcn.sgpr, !amdgcn.sgpr
//       CHECK:   %[[FLAGS_VAL2:.*]] = sop1 s_mov_b32 outs %{{.*}} ins %c131076_i32 : !amdgcn.sgpr, i32
//       CHECK:   %[[RSRC2:.*]] = make_register_range %[[DWORD0_2]], %[[DWORD1]], %[[NREC2]], %[[FLAGS_VAL2]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
//       CHECK:   test_inst ins %[[RSRC2]] : (!amdgcn.sgpr_range<[? + 4]>) -> ()
//       CHECK:   end_kernel
  kernel @make_buffer_rsrc_structured arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = read_only>,
      #amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>
    ]> {
    %base = load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %num_records = load_arg 1 : !amdgcn.sgpr
    // stride=16 (float4), flags encodes dst_sel and data_format
    %stride = arith.constant 16 : i32
    %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %stride,
      cache_swizzle = false, swizzle_enable = false, flags = 131076
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
    test_inst ins %rsrc : (!amdgcn.sgpr_range<[? + 4]>) -> ()
    end_kernel
  }

// stride=4, cache_swizzle=true, swizzle_enable=true
// dword1 upper = (4 << 16) | (1 << 30) | (1 << 31) = 262144 + 1073741824 + 2147483648 = 3221487616
// as signed i32: -1073479680
// CHECK-LABEL: kernel @make_buffer_rsrc_swizzled
//       CHECK:   %[[BASE3:.*]], %{{.*}} = load s_load_dwordx2
//       CHECK:   %[[NREC3:.*]], %{{.*}} = load s_load_dword
//       CHECK:   %[[BASE_SPLIT3:.*]]:2 = split_register_range %[[BASE3]] : !amdgcn.sgpr_range<[? + 2]>
//       CHECK:   %[[DWORD1_3:.*]] = sop2 s_or_b32 outs %{{.*}} ins %[[BASE_SPLIT3]]#1, %c-1073479680_i32 : !amdgcn.sgpr, !amdgcn.sgpr, i32
//       CHECK:   %[[DWORD0_3:.*]] = sop1 s_mov_b32 outs %{{.*}} ins %[[BASE_SPLIT3]]#0 : !amdgcn.sgpr, !amdgcn.sgpr
//       CHECK:   %[[FLAGS_VAL3:.*]] = sop1 s_mov_b32 outs %{{.*}} ins %c0_i32 : !amdgcn.sgpr, i32
//       CHECK:   %[[RSRC3:.*]] = make_register_range %[[DWORD0_3]], %[[DWORD1_3]], %[[NREC3]], %[[FLAGS_VAL3]] : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
//       CHECK:   test_inst ins %[[RSRC3]] : (!amdgcn.sgpr_range<[? + 4]>) -> ()
//       CHECK:   end_kernel
  kernel @make_buffer_rsrc_swizzled arguments <[
      #amdgcn.buffer_arg<address_space = generic, access = read_only>,
      #amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>
    ]> {
    %base = load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %num_records = load_arg 1 : !amdgcn.sgpr
    // stride=4 (f32), both swizzle modes enabled
    %stride = arith.constant 4 : i32
    %rsrc = amdgcn.make_buffer_rsrc %base, %num_records, %stride,
      cache_swizzle = true, swizzle_enable = true, flags = 0
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
    test_inst ins %rsrc : (!amdgcn.sgpr_range<[? + 4]>) -> ()
    end_kernel
  }
}
