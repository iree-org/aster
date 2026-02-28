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
    %tok_store = amdgcn.store global_store_dwordx4 data %data_range addr %addr_range : ins(!amdgcn.vgpr<[0 : 4]>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>

    // Write to VGPRs that overlap with the store's data VGPRs (should trigger case 8)
    // Writing to %data1 (VGPR 1) which is in the store's data range [0:4)
    amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %data1, %data0
      : (!amdgcn.vgpr<1>, !amdgcn.vgpr<0>) -> ()

    amdgcn.end_kernel
  }
}

// -----

// CHECK-LABEL: kernel @test_kernel
//       CHECK:   amdgcn.sopp.sopp <s_nop>, imm = 5
//  CHECK-NEXT:   load global_load_dword
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
    amdgcn.vop2 v_add_co_u32 outs %data0 dst1 = %sgpr_carry ins %data0, %data0
      : !amdgcn.vgpr<0>, !amdgcn.sgpr<[0 : 2]>, !amdgcn.vgpr<0>, !amdgcn.vgpr<0>

    // Use the SGPR carry result in the address range
    // Split the carry range to get individual SGPRs (returns 2 results for size 2)
    %carry0, %carry1 = amdgcn.split_register_range %sgpr_carry : !amdgcn.sgpr<[0 : 2]>
    %addr_range = amdgcn.make_register_range %carry0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>

    // VMEM instruction (global_load) that reads from the SGPR written by VALU
    // This should trigger case 10 (requires 5 NOPs)
    %dst_range = amdgcn.make_register_range %result0 : !amdgcn.vgpr<1>
    %tok_load = amdgcn.load global_load_dword dest %dst_range addr %addr_range : dps(!amdgcn.vgpr<1>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<flat>

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
    %tok_store = amdgcn.store global_store_dwordx3 data %data_range addr %addr_range : ins(!amdgcn.vgpr<[0 : 3]>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>

    // Write to different VGPRs (no overlap) - should NOT trigger case 8
    // Writing to registers 10-11 which don't overlap with store data range [0:3)
    amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %result0, %result1
      : (!amdgcn.vgpr<10>, !amdgcn.vgpr<11>) -> ()

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
    %tok_store = amdgcn.store global_store_dwordx3 data %data_range addr %addr_range : ins(!amdgcn.vgpr<[0 : 3]>, !amdgcn.sgpr<[0 : 2]>) -> !amdgcn.write_token<flat>

    // VALU instruction writing to different VGPRs (no overlap) - should NOT trigger case 9
    // Writing to registers 10-11 which don't overlap with store data range [0:3)
    amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %result0, %result1
      : (!amdgcn.vgpr<10>, !amdgcn.vgpr<11>) -> ()

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
    amdgcn.vop2 v_add_co_u32 outs %data0 dst1 = %sgpr_carry ins %data0, %data0
      : !amdgcn.vgpr<0>, !amdgcn.sgpr<[10 : 12]>, !amdgcn.vgpr<0>, !amdgcn.vgpr<0>

    // VMEM instruction reads from different SGPRs (no overlap) - should NOT trigger case 10
    // Reading from SGPRs 0-1, but VALU wrote to SGPR 10
    %addr_range = amdgcn.make_register_range %addr0, %addr1 : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>
    %dst_range = amdgcn.make_register_range %result0 : !amdgcn.vgpr<1>
    %tok_load = amdgcn.load global_load_dword dest %dst_range addr %addr_range : dps(!amdgcn.vgpr<1>) ins(!amdgcn.sgpr<[0 : 2]>) -> !amdgcn.read_token<flat>

    amdgcn.end_kernel
  }
}

// -----

// Case 106 for CDNA4 scaled MFMA 16x16x128 (2-pass): MFMA write VGPR -> VALU read/write overlapping vDst
// Expects 7 v_nop insertions (conservative for 2-pass)
// CHECK-LABEL: kernel @test_kernel
//       CHECK:   amdgcn.vop3p.vop3p_scaled_mai
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//  CHECK-NEXT:   v_mov_b32_e32
amdgcn.module @test_case106_scaled_mfma_16x16x128 target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {
  amdgcn.kernel @test_kernel {
    // A operands: 8 VGPRs [0:8)
    %a0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %a1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %a2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %a3 = amdgcn.alloca : !amdgcn.vgpr<3>
    %a4 = amdgcn.alloca : !amdgcn.vgpr<4>
    %a5 = amdgcn.alloca : !amdgcn.vgpr<5>
    %a6 = amdgcn.alloca : !amdgcn.vgpr<6>
    %a7 = amdgcn.alloca : !amdgcn.vgpr<7>
    %a_range = amdgcn.make_register_range %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7
      : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>,
        !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>

    // B operands: 8 VGPRs [8:16)
    %b0 = amdgcn.alloca : !amdgcn.vgpr<8>
    %b1 = amdgcn.alloca : !amdgcn.vgpr<9>
    %b2 = amdgcn.alloca : !amdgcn.vgpr<10>
    %b3 = amdgcn.alloca : !amdgcn.vgpr<11>
    %b4 = amdgcn.alloca : !amdgcn.vgpr<12>
    %b5 = amdgcn.alloca : !amdgcn.vgpr<13>
    %b6 = amdgcn.alloca : !amdgcn.vgpr<14>
    %b7 = amdgcn.alloca : !amdgcn.vgpr<15>
    %b_range = amdgcn.make_register_range %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7
      : !amdgcn.vgpr<8>, !amdgcn.vgpr<9>, !amdgcn.vgpr<10>, !amdgcn.vgpr<11>,
        !amdgcn.vgpr<12>, !amdgcn.vgpr<13>, !amdgcn.vgpr<14>, !amdgcn.vgpr<15>

    // C/dst operands: 4 VGPRs [16:20)
    %c0 = amdgcn.alloca : !amdgcn.vgpr<16>
    %c1 = amdgcn.alloca : !amdgcn.vgpr<17>
    %c2 = amdgcn.alloca : !amdgcn.vgpr<18>
    %c3 = amdgcn.alloca : !amdgcn.vgpr<19>
    %c_range = amdgcn.make_register_range %c0, %c1, %c2, %c3
      : !amdgcn.vgpr<16>, !amdgcn.vgpr<17>, !amdgcn.vgpr<18>, !amdgcn.vgpr<19>

    // Scale sources
    %s0 = amdgcn.alloca : !amdgcn.vgpr<32>
    %s1 = amdgcn.alloca : !amdgcn.vgpr<33>

    // Scaled MFMA 16x16x128: writes to VGPRs [16:20)
    amdgcn.vop3p.vop3p_scaled_mai <v_mfma_scale_f32_16x16x128_f8f6f4>
      %c_range, %a_range, %b_range, %c_range, %s0, %s1
      : <[0 : 8]>, <[8 : 16]>, !amdgcn.vgpr<[16 : 20]>, !amdgcn.vgpr<32>, !amdgcn.vgpr<33>
      -> !amdgcn.vgpr<[16 : 20]>

    // VALU reads from overlapping VGPR (v16) -> triggers case 106
    amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %c0, %c0
      : (!amdgcn.vgpr<16>, !amdgcn.vgpr<16>) -> ()

    amdgcn.end_kernel
  }
}

// -----

// Case 106 for CDNA4 scaled MFMA 32x32x64 (4-pass): MFMA write VGPR -> VALU read/write overlapping vDst
// Expects 7 v_nop insertions (4-pass)
// CHECK-LABEL: kernel @test_kernel
//       CHECK:   amdgcn.vop3p.vop3p_scaled_mai
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//  CHECK-NEXT:   v_mov_b32_e32
amdgcn.module @test_case106_scaled_mfma_32x32x64 target = #amdgcn.target<gfx950> isa = #amdgcn.isa<cdna4> {
  amdgcn.kernel @test_kernel {
    // A operands: 8 VGPRs [0:8)
    %a0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %a1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %a2 = amdgcn.alloca : !amdgcn.vgpr<2>
    %a3 = amdgcn.alloca : !amdgcn.vgpr<3>
    %a4 = amdgcn.alloca : !amdgcn.vgpr<4>
    %a5 = amdgcn.alloca : !amdgcn.vgpr<5>
    %a6 = amdgcn.alloca : !amdgcn.vgpr<6>
    %a7 = amdgcn.alloca : !amdgcn.vgpr<7>
    %a_range = amdgcn.make_register_range %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7
      : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>, !amdgcn.vgpr<2>, !amdgcn.vgpr<3>,
        !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>

    // B operands: 8 VGPRs [8:16)
    %b0 = amdgcn.alloca : !amdgcn.vgpr<8>
    %b1 = amdgcn.alloca : !amdgcn.vgpr<9>
    %b2 = amdgcn.alloca : !amdgcn.vgpr<10>
    %b3 = amdgcn.alloca : !amdgcn.vgpr<11>
    %b4 = amdgcn.alloca : !amdgcn.vgpr<12>
    %b5 = amdgcn.alloca : !amdgcn.vgpr<13>
    %b6 = amdgcn.alloca : !amdgcn.vgpr<14>
    %b7 = amdgcn.alloca : !amdgcn.vgpr<15>
    %b_range = amdgcn.make_register_range %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7
      : !amdgcn.vgpr<8>, !amdgcn.vgpr<9>, !amdgcn.vgpr<10>, !amdgcn.vgpr<11>,
        !amdgcn.vgpr<12>, !amdgcn.vgpr<13>, !amdgcn.vgpr<14>, !amdgcn.vgpr<15>

    // C/dst operands: 16 VGPRs [16:32)
    %c0 = amdgcn.alloca : !amdgcn.vgpr<16>
    %c1 = amdgcn.alloca : !amdgcn.vgpr<17>
    %c2 = amdgcn.alloca : !amdgcn.vgpr<18>
    %c3 = amdgcn.alloca : !amdgcn.vgpr<19>
    %c4 = amdgcn.alloca : !amdgcn.vgpr<20>
    %c5 = amdgcn.alloca : !amdgcn.vgpr<21>
    %c6 = amdgcn.alloca : !amdgcn.vgpr<22>
    %c7 = amdgcn.alloca : !amdgcn.vgpr<23>
    %c8 = amdgcn.alloca : !amdgcn.vgpr<24>
    %c9 = amdgcn.alloca : !amdgcn.vgpr<25>
    %c10 = amdgcn.alloca : !amdgcn.vgpr<26>
    %c11 = amdgcn.alloca : !amdgcn.vgpr<27>
    %c12 = amdgcn.alloca : !amdgcn.vgpr<28>
    %c13 = amdgcn.alloca : !amdgcn.vgpr<29>
    %c14 = amdgcn.alloca : !amdgcn.vgpr<30>
    %c15 = amdgcn.alloca : !amdgcn.vgpr<31>
    %c_range = amdgcn.make_register_range %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7,
      %c8, %c9, %c10, %c11, %c12, %c13, %c14, %c15
      : !amdgcn.vgpr<16>, !amdgcn.vgpr<17>, !amdgcn.vgpr<18>, !amdgcn.vgpr<19>,
        !amdgcn.vgpr<20>, !amdgcn.vgpr<21>, !amdgcn.vgpr<22>, !amdgcn.vgpr<23>,
        !amdgcn.vgpr<24>, !amdgcn.vgpr<25>, !amdgcn.vgpr<26>, !amdgcn.vgpr<27>,
        !amdgcn.vgpr<28>, !amdgcn.vgpr<29>, !amdgcn.vgpr<30>, !amdgcn.vgpr<31>

    // Scale sources
    %s0 = amdgcn.alloca : !amdgcn.vgpr<32>
    %s1 = amdgcn.alloca : !amdgcn.vgpr<33>

    // Scaled MFMA 32x32x64: writes to VGPRs [16:32)
    amdgcn.vop3p.vop3p_scaled_mai <v_mfma_scale_f32_32x32x64_f8f6f4>
      %c_range, %a_range, %b_range, %c_range, %s0, %s1
      : <[0 : 8]>, <[8 : 16]>, !amdgcn.vgpr<[16 : 32]>, !amdgcn.vgpr<32>, !amdgcn.vgpr<33>
      -> !amdgcn.vgpr<[16 : 32]>

    // VALU reads from overlapping VGPR (v16) -> triggers case 106
    amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %c0, %c0
      : (!amdgcn.vgpr<16>, !amdgcn.vgpr<16>) -> ()

    amdgcn.end_kernel
  }
}

// -----

// Case 106 for CDNA3 FP8 MFMA 16x16x32 (2-pass): MFMA write VGPR -> VALU read/write overlapping vDst
// Expects 7 v_nop insertions (conservative for 2-pass)
// CHECK-LABEL: kernel @test_kernel
//       CHECK:   amdgcn.vop3p.vop3p_mai
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//       CHECK:   amdgcn.vop1.v_nop
//  CHECK-NEXT:   v_mov_b32_e32
amdgcn.module @test_case106_fp8_mfma_16x16x32 target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_kernel {
    // A operands: 2 VGPRs [0:2)
    %a0 = amdgcn.alloca : !amdgcn.vgpr<0>
    %a1 = amdgcn.alloca : !amdgcn.vgpr<1>
    %a_range = amdgcn.make_register_range %a0, %a1
      : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>

    // B operands: 2 VGPRs [2:4)
    %b0 = amdgcn.alloca : !amdgcn.vgpr<2>
    %b1 = amdgcn.alloca : !amdgcn.vgpr<3>
    %b_range = amdgcn.make_register_range %b0, %b1
      : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>

    // C/D operands: 4 VGPRs [4:8)
    %c0 = amdgcn.alloca : !amdgcn.vgpr<4>
    %c1 = amdgcn.alloca : !amdgcn.vgpr<5>
    %c2 = amdgcn.alloca : !amdgcn.vgpr<6>
    %c3 = amdgcn.alloca : !amdgcn.vgpr<7>
    %c_range = amdgcn.make_register_range %c0, %c1, %c2, %c3
      : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>

    // FP8 MFMA 16x16x32: writes to VGPRs [4:8)
    amdgcn.vop3p.vop3p_mai <v_mfma_f32_16x16x32_fp8_fp8>
      %c_range, %a_range, %b_range, %c_range
      : <[0 : 2]>, <[2 : 4]>, !amdgcn.vgpr<[4 : 8]> -> !amdgcn.vgpr<[4 : 8]>

    // VALU reads from overlapping VGPR (v4) -> triggers case 106
    amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %c0, %c0
      : (!amdgcn.vgpr<4>, !amdgcn.vgpr<4>) -> ()

    amdgcn.end_kernel
  }
}
