// RUN: aster-opt %s --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(amdgcn-nop-insertion)))" --split-input-file 2>&1 | FileCheck %s

// Case 8: FLAT_STORE_X4 followed by write to same VGPRs (requires 1 NOP)
// Case 9: FLAT_STORE_X4 followed by VALU write to same VGPRs (requires 2 NOPs)
// CHECK-LABEL: kernel @test_kernel
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   v_mov_b32_e32
amdgcn.module @test_case8_9_store_x4 target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %data1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %data2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %data3 = amdgcn.alloca : !amdgcn.vgpr<3>
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>

    %addr_range = amdgcn.make_register_range %addr0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %data_range = amdgcn.make_register_range %data0, %data1, %data2, %data3 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>

    // FLAT_STORE_X4 followed by write to same VGPRs
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

// Case 10: VALU writes SGPR -> VMEM reads that SGPR (requires 5 NOPs)
// CHECK-LABEL: kernel @test_kernel
//       CHECK:   amdgcn.sopp.sopp <s_nop>, imm = 5
//       CHECK:   amdgcn.flat.global_load <global_load_dword>
amdgcn.module @test_case10_valu_sgpr_vmem target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %result0 = amdgcn.alloca : !amdgcn.vgpr<1>

    // VALU writes SGPR -> VMEM reads that SGPR
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

// Case 100: Non-DLops VALU Write VGPR -> V_MFMA* read VGPR (requires 2 NOPs)
// CHECK-LABEL: kernel @test_kernel
//       CHECK:   v_mov_b32_e32
// CHECK-COUNT-2: v_nop
//       CHECK:   vop3p_mai
amdgcn.module @test_case100_valu_to_mfma target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %vgpr_a0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %vgpr_a1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %vgpr_b0 = amdgcn.alloca : !amdgcn.vgpr<2>
    %vgpr_b1 = amdgcn.alloca : !amdgcn.vgpr<3>
    %agpr0 = amdgcn.alloca : !amdgcn.agpr<0>
    %agpr1 = amdgcn.alloca : !amdgcn.agpr<1>
    %agpr2 = amdgcn.alloca : !amdgcn.agpr<2>
    %agpr3 = amdgcn.alloca : !amdgcn.agpr<3>

    // Non-DLops VALU writes to VGPR 0
    %0 = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %vgpr_a0, %vgpr_b0
      : (!amdgcn.vgpr<0>, !amdgcn.vgpr<2>) -> !amdgcn.vgpr<0>

    // MFMA reads VGPR 0 as SrcA - should trigger Case 100 (2 NOPs)
    %a_range = amdgcn.make_register_range %vgpr_a0, %vgpr_a1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
    %b_range = amdgcn.make_register_range %vgpr_b0, %vgpr_b1 : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
    %dst_agpr_range = amdgcn.make_register_range %agpr0, %agpr1, %agpr2, %agpr3 : !amdgcn.agpr<0>, !amdgcn.agpr<1>, !amdgcn.agpr<2>, !amdgcn.agpr<3>

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst_agpr_range, %a_range, %b_range, %dst_agpr_range
      : !amdgcn.vgpr_range<[0 : 2]>, !amdgcn.vgpr_range<[2 : 4]>, !amdgcn.agpr_range<[0 : 4]>
    -> !amdgcn.agpr_range<[0 : 4]>

    amdgcn.end_kernel
  }
}

// -----

// Case 106: v_accvgpr_read_b32 reads from AGPR written by vop3p_mai (requires 7 NOPs)
// CHECK-LABEL: kernel @test_kernel
//       CHECK:   vop3p_mai
// CHECK-COUNT-7: v_nop
//       CHECK:   v_accvgpr_read_b32
amdgcn.module @test_vop3p_mai_agpr target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %agpr0 = amdgcn.alloca : !amdgcn.agpr<0>
    %agpr1 = amdgcn.alloca : !amdgcn.agpr<1>
    %agpr2 = amdgcn.alloca : !amdgcn.agpr<2>
    %agpr3 = amdgcn.alloca : !amdgcn.agpr<3>
    %vgpr_a0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %vgpr_a1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %vgpr_b0 = amdgcn.alloca : !amdgcn.vgpr<2>
    %vgpr_b1 = amdgcn.alloca : !amdgcn.vgpr<3>
    %vgpr_result = amdgcn.alloca : !amdgcn.vgpr<8>

    %a_range = amdgcn.make_register_range %vgpr_a0, %vgpr_a1 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
    %b_range = amdgcn.make_register_range %vgpr_b0, %vgpr_b1 : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
    %dst_agpr_range = amdgcn.make_register_range %agpr0, %agpr1, %agpr2, %agpr3 : !amdgcn.agpr<0>, !amdgcn.agpr<1>, !amdgcn.agpr<2>, !amdgcn.agpr<3>

    %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst_agpr_range, %a_range, %b_range, %dst_agpr_range
      : !amdgcn.vgpr_range<[0 : 2]>, !amdgcn.vgpr_range<[2 : 4]>, !amdgcn.agpr_range<[0 : 4]>
    -> !amdgcn.agpr_range<[0 : 4]>

    %0 = amdgcn.vop3p v_accvgpr_read_b32 outs %vgpr_result ins %agpr0 : !amdgcn.vgpr<8>, !amdgcn.agpr<0>

    amdgcn.end_kernel
  }
}


//==-------------------------------------------------------------------------==//
// -----
// NEGATIVE TESTS START HERE -----
//==-------------------------------------------------------------------------==//

// Case 8 (negative): No overlap - should NOT trigger case 8
// CHECK-LABEL: kernel @test_kernel
//   CHECK-NOT:   s_nop
amdgcn.module @test_case8_no_overlap target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %data1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %data2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %result0 = amdgcn.alloca : !amdgcn.vgpr<10>
    %result1 = amdgcn.alloca : !amdgcn.vgpr<11>

    %addr_range = amdgcn.make_register_range %addr0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %data_range = amdgcn.make_register_range %data0, %data1, %data2 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>

    // FLAT_STORE_X3
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

// Case 9 (negative): No overlap - should NOT trigger case 9
// CHECK-LABEL: kernel @test_kernel
//   CHECK-NOT:   s_nop
amdgcn.module @test_case9_no_overlap_valu target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    %data0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %data1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %data2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %addr0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %addr1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %result0 = amdgcn.alloca : !amdgcn.vgpr<10>
    %result1 = amdgcn.alloca : !amdgcn.vgpr<11>

    %addr_range = amdgcn.make_register_range %addr0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %data_range = amdgcn.make_register_range %data0, %data1, %data2 : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>

    // FLAT_STORE_X3
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

// Case 10 (negative): No overlap - should NOT trigger case 10
// CHECK-LABEL: kernel @test_kernel
//   CHECK-NOT:   s_nop
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

    // VALU writes SGPR
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
