// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops, aster-hoist-ops, canonicalize, aster-amdgcn-expand-md-ops)))" | FileCheck %s
//
// Test that running aster-amdgcn-expand-md-ops twice does not clobber kernel
// attributes set by the first run. Each attribute group (workgroup_id enables,
// workitem_id_mode) is only modified when the corresponding ops are present,
// so the second run (with all ops already expanded) preserves everything.

amdgcn.module @idempotent_test target = <gfx942> {

// block_id x + y + z: workgroup_id enables must survive second run.
// CHECK-LABEL: kernel @bid_xyz
// CHECK-SAME:    enable_workgroup_id_y
// CHECK-SAME:    enable_workgroup_id_z
  kernel @bid_xyz arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = block_id  x : !amdgcn.sgpr
    %1 = block_id  y : !amdgcn.sgpr
    %2 = block_id  z : !amdgcn.sgpr
    test_inst ins %0, %1, %2 : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }

// thread_id x + y: workitem_id_mode must survive second run.
// CHECK-LABEL: kernel @tid_xy
// CHECK-SAME:    enable_workgroup_id_x = false
// CHECK-SAME:    workitem_id_mode = #amdgcn.workitem_id_mode<x_y>
  kernel @tid_xy arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  x : !amdgcn.vgpr
    %1 = thread_id  y : !amdgcn.vgpr
    test_inst ins %0, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }

// Both thread_id and block_id: both attribute groups must survive.
// CHECK-LABEL: kernel @tid_bid_mixed
// CHECK-SAME:    enable_workgroup_id_y
// CHECK-SAME:    workitem_id_mode = #amdgcn.workitem_id_mode<x_y_z>
  kernel @tid_bid_mixed arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  x : !amdgcn.vgpr
    %1 = thread_id  y : !amdgcn.vgpr
    %2 = thread_id  z : !amdgcn.vgpr
    %3 = block_id  x : !amdgcn.sgpr
    %4 = block_id  y : !amdgcn.sgpr
    test_inst ins %0, %1, %2, %3, %4 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }

// No metadata ops: neither attribute group modified.
// CHECK-LABEL: kernel @no_metadata
// CHECK-NOT:     enable_workgroup_id
// CHECK-NOT:     workitem_id_mode
// CHECK:         end_kernel
  kernel @no_metadata arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    end_kernel
  }
}
