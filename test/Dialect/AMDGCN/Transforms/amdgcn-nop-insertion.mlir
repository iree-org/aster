// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-nop-insertion)))" --split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: kernel @test_kernel
//       case 8:
//       CHECK:   amdgcn.vop1.v_nop
//       case 9:
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//  CHECK-NEXT:   v_mov_b32_e32
amdgcn.module @test_case8_9_store_x4 target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    // Allocate VGPRs for data (4 VGPRs for dwordx4) - using registers 0-3
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %data1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %data2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %data3 = amdgcn.alloca : !amdgcn.vgpr<3>
    // Allocate SGPRs for address (2 SGPRs) - using registers 0-1
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>

    %addr_range = amdgcn.make_register_range %addr0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %data_range = amdgcn.make_register_range %data0, %data1, %data2, %data3 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>

    // Case 8: FLAT_STORE_X4 followed by write to same VGPRs
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx4> %data_range, %addr_range, offset = 0
      : !amdgcn.vgpr_range<[0 : 4]>, !amdgcn.sgpr_range<[0 : 2]>

    // Write to VGPRs that overlap with the store's data VGPRs (should trigger case 8)
    // Writing to %data1 (VGPR 1) which is in the store's data range [0:4)
    %0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %data1, %data0
      : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> !amdgcn.vgpr<1>

    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: kernel @test_kernel
//       CHECK:   amdgcn.sopp.sopp <s_nop>, imm = 5
//  CHECK-NEXT:   amdgcn.flat.global_load <global_load_dword>
amdgcn.module @test_case10_valu_sgpr_vmem target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    // Allocate SGPRs for address (2 SGPRs) - using registers 0-1
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>
    // Allocate VGPRs for data and result
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %result0 = amdgcn.alloca : !amdgcn.vgpr<1>

    // Case 10: VALU writes SGPR -> VMEM reads that SGPR
    // VALU instruction (v_add_co_u32) that writes to SGPR (carry/VCC output)
    // v_add_co_u32 writes to both VGPR (vdst0) and SGPR (dst1/carry)
    // Allocate individual SGPRs for carry (VCC is 2 SGPRs)
    %carry0_sgpr = amdgcn.alloca : !amdgcn.sgpr<0>
    %carry1_sgpr = amdgcn.alloca : !amdgcn.sgpr<1>
    %sgpr_carry = amdgcn.make_register_range %carry0_sgpr, %carry1_sgpr : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %vgpr_result, %sgpr_result = amdgcn.vop2 v_add_co_u32 outs %data0 dst1 = %sgpr_carry ins %data0, %data0
      : !amdgcn.vgpr<0>, !amdgcn.sgpr_range<[0 : 2]>, !amdgcn.vgpr<0>, !amdgcn.vgpr<0>

    // Use the SGPR carry result in the address range
    // Split the carry range to get individual SGPRs (returns 2 results for size 2)
    %carry0, %carry1 = amdgcn.split_register_range %sgpr_result : !amdgcn.sgpr_range<[0 : 2]>
    %addr_range = amdgcn.make_register_range %carry0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>

    // VMEM instruction (global_load) that reads from the SGPR written by VALU
    // This should trigger case 10 (requires 5 NOPs)
    %dst_range = amdgcn.make_register_range %result0 : !amdgcn.vgpr<1>
    %0 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range, %addr_range, offset = 0
      : !amdgcn.vgpr_range<[1 : 2]>, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.vgpr_range<[1 : 2]>

    amdgcn.end_kernel
  }
}


// -----

//
// NEGATIVE TESTS START HERE -----
//

// CHECK-LABEL: kernel @test_kernel
//   CHECK-NOT:  s_nop
amdgcn.module @test_case8_no_overlap target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    // Allocate VGPRs for data (3 VGPRs for dwordx3) - using registers 0-2
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %data1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %data2 = amdgcn.alloca : !amdgcn.vgpr<2>
    // Allocate SGPRs for address (2 SGPRs) - using registers 0-1
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>
    // Allocate different VGPRs for result (no overlap) - using registers 10-11
    %result0 = amdgcn.alloca : !amdgcn.vgpr<10>
    %result1 = amdgcn.alloca : !amdgcn.vgpr<11>

    %addr_range = amdgcn.make_register_range %addr0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %data_range = amdgcn.make_register_range %data0, %data1, %data2 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>

    // Case 8: FLAT_STORE_X3
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx3> %data_range, %addr_range, offset = 0
      : !amdgcn.vgpr_range<[0 : 3]>, !amdgcn.sgpr_range<[0 : 2]>

    // Write to different VGPRs (no overlap) - should NOT trigger case 8
    // Writing to registers 10-11 which don't overlap with store data range [0:3)
    %0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %result0, %result1
      : (!amdgcn.vgpr<10>, !amdgcn.vgpr<11>) -> !amdgcn.vgpr<10>

    amdgcn.end_kernel
  }
}


// -----

// CHECK-LABEL: kernel @test_kernel
//   CHECK-NOT:  s_nop
amdgcn.module @test_case9_no_overlap_valu target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    // Allocate VGPRs for data (3 VGPRs for dwordx3) - using registers 0-2
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %data1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %data2 = amdgcn.alloca : !amdgcn.vgpr<2>
    // Allocate SGPRs for address (2 SGPRs) - using registers 0-1
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>
    // Allocate different VGPRs for result (no overlap) - using registers 10-11
    %result0 = amdgcn.alloca : !amdgcn.vgpr<10>
    %result1 = amdgcn.alloca : !amdgcn.vgpr<11>

    %addr_range = amdgcn.make_register_range %addr0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %data_range = amdgcn.make_register_range %data0, %data1, %data2 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>

    // Case 9: FLAT_STORE_X3
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx3> %data_range, %addr_range, offset = 0
      : !amdgcn.vgpr_range<[0 : 3]>, !amdgcn.sgpr_range<[0 : 2]>

    // VALU instruction writing to different VGPRs (no overlap) - should NOT trigger case 9
    // Writing to registers 10-11 which don't overlap with store data range [0:3)
    %0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %result0, %result1
      : (!amdgcn.vgpr<10>, !amdgcn.vgpr<11>) -> !amdgcn.vgpr<10>

    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: kernel @test_kernel
//   CHECK-NOT:  s_nop
amdgcn.module @test_case10_no_overlap_sgpr target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    // Allocate SGPRs for address (2 SGPRs) - using registers 0-1
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>
    // Allocate different SGPRs for VALU result (no overlap) - using register 10
    %sgpr_result = amdgcn.alloca : !amdgcn.sgpr<10>
    // Allocate VGPRs for data and result
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %result0 = amdgcn.alloca : !amdgcn.vgpr<1>

    // Case 10: VALU writes SGPR
    // v_add_co_u32 writes to both VGPR and SGPR (carry)
    // Allocate individual SGPRs for carry (VCC is 2 SGPRs) - using registers 10-11
    %carry0 = amdgcn.alloca : !amdgcn.sgpr<10>
    %carry1 = amdgcn.alloca : !amdgcn.sgpr<11>
    %sgpr_carry = amdgcn.make_register_range %carry0, %carry1 : !amdgcn.sgpr<10>, !amdgcn.sgpr<11>
    %0, %carry_out = amdgcn.vop2 v_add_co_u32 outs %data0 dst1 = %sgpr_carry ins %data0, %data0
      : !amdgcn.vgpr<0>, !amdgcn.sgpr_range<[10 : 12]>, !amdgcn.vgpr<0>, !amdgcn.vgpr<0>

    // VMEM instruction reads from different SGPRs (no overlap) - should NOT trigger case 10
    // Reading from SGPRs 0-1, but VALU wrote to SGPR 10
    %addr_range = amdgcn.make_register_range %addr0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %dst_range = amdgcn.make_register_range %result0 : !amdgcn.vgpr<1>
    %2 = amdgcn.flat.global_load #amdgcn.inst<global_load_dword> %dst_range, %addr_range, offset = 0
      : !amdgcn.vgpr_range<[1 : 2]>, !amdgcn.sgpr_range<[0 : 2]> -> !amdgcn.vgpr_range<[1 : 2]>

    amdgcn.end_kernel
  }
}
