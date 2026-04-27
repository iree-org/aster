// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops{force-unpacked-tid=true}, aster-hoist-ops, canonicalize)))" | FileCheck %s
//
// Test the unpacked workitem ID path (legacy CDNA1/CDNA2/GFX9 convention):
//   X = VGPR0, Y = VGPR1, Z = VGPR2 (each in its own register).
//
// This path is selected by force-unpacked-tid=true or by targeting an ISA
// without FeaturePackedTID.

amdgcn.module @unpacked_tid_test target = <gfx942> {

// Thread X only (unpacked): X comes directly from VGPR0.
// Same as packed when only X is used (no masking needed in either case).
// CHECK-LABEL: kernel @thread_x_unpacked
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK:         test_inst ins %[[V0]] : (!amdgcn.vgpr<0>) -> ()
// CHECK:         end_kernel
  kernel @thread_x_unpacked arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  x : !amdgcn.vgpr
    test_inst ins %0 : (!amdgcn.vgpr) -> ()
    end_kernel
  }

// Thread X + Y (unpacked): X from VGPR0, Y from VGPR1.
// No shift/mask needed -- each dimension is in its own VGPR.
// CHECK-LABEL: kernel @thread_xy_unpacked
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:         test_inst ins %[[V0]], %[[V1]] : (!amdgcn.vgpr<0>, !amdgcn.vgpr<1>) -> ()
// CHECK:         end_kernel
  kernel @thread_xy_unpacked arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  x : !amdgcn.vgpr
    %1 = thread_id  y : !amdgcn.vgpr
    test_inst ins %0, %1 : (!amdgcn.vgpr, !amdgcn.vgpr) -> ()
    end_kernel
  }

// Thread X + Y + Z (unpacked): X from VGPR0, Y from VGPR1, Z from VGPR2.
// No shift/mask needed -- each dimension is in its own VGPR.
// In contrast, the packed path would extract all from VGPR0 via shift+mask.
// CHECK-LABEL: kernel @thread_xyz_unpacked
// CHECK-DAG:     %[[V0:.*]] = alloca : !amdgcn.vgpr<0>
// CHECK-DAG:     %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK-DAG:     %[[V2:.*]] = alloca : !amdgcn.vgpr<2>
// CHECK-DAG:     %[[BID_X:.*]] = alloca : !amdgcn.sgpr<2>
// CHECK-DAG:     %[[BID_Y:.*]] = alloca : !amdgcn.sgpr<3>
// CHECK-DAG:     %[[BID_Z:.*]] = alloca : !amdgcn.sgpr<4>
// CHECK:         test_inst ins %[[V0]], %[[V1]], %[[V2]], %[[BID_X]], %[[BID_Y]], %[[BID_Z]]
// CHECK:         end_kernel
  kernel @thread_xyz_unpacked arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  x : !amdgcn.vgpr
    %1 = thread_id  y : !amdgcn.vgpr
    %2 = thread_id  z : !amdgcn.vgpr
    %3 = block_id  x : !amdgcn.sgpr
    %4 = block_id  y : !amdgcn.sgpr
    %5 = block_id  z : !amdgcn.sgpr
    test_inst ins %0, %1, %2, %3, %4, %5 : (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> ()
    end_kernel
  }

// Verify workitem_id_mode is still set correctly for unpacked path.
// The mode tells the hardware which VGPRs to initialize (same for both paths).
// CHECK-LABEL: kernel @thread_y_only_unpacked
// CHECK-SAME:    workitem_id_mode = #amdgcn.workitem_id_mode<x_y>
// CHECK-DAG:     %[[V1:.*]] = alloca : !amdgcn.vgpr<1>
// CHECK:         test_inst ins %[[V1]] : (!amdgcn.vgpr<1>) -> ()
// CHECK:         end_kernel
  kernel @thread_y_only_unpacked arguments <[#amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>]> {
    %0 = thread_id  y : !amdgcn.vgpr
    test_inst ins %0 : (!amdgcn.vgpr) -> ()
    end_kernel
  }
}
