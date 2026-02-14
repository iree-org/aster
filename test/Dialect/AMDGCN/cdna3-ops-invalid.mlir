// RUN: aster-opt %s --verify-diagnostics --split-input-file

//===----------------------------------------------------------------------===//
// CDNA3 VOP3P_MAI Register Count Verification
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong A operand register count (3 instead of 2)

func.func @test_vop3p_mai_wrong_a_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 3]>, %b: !amdgcn.vgpr<[4 : 6]>, %c: !amdgcn.vgpr<[8 : 12]>) {
  // expected-error@+1 {{a operand must have 2 registers for v_mfma_f32_16x16x16_f16, got 3}}
  amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %a, %b, %c
      : !amdgcn.vgpr<[0 : 3]>, !amdgcn.vgpr<[4 : 6]>, !amdgcn.vgpr<[8 : 12]>
    -> !amdgcn.vgpr<[12 : 16]>
  return
}

// -----
// Test: Wrong B operand register count (3 instead of 2)

func.func @test_vop3p_mai_wrong_b_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[4 : 7]>, %c: !amdgcn.vgpr<[8 : 12]>) {
  // expected-error@+1 {{b operand must have 2 registers for v_mfma_f32_16x16x16_f16, got 3}}
  amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %a, %b, %c
      : !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[4 : 7]>, !amdgcn.vgpr<[8 : 12]>
    -> !amdgcn.vgpr<[12 : 16]>
  return
}

// -----
// Test: Wrong C operand register count (2 instead of 4)

func.func @test_vop3p_mai_wrong_c_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.vgpr<[4 : 6]>) {
  // expected-error@+1 {{c operand must have 4 registers for v_mfma_f32_16x16x16_f16, got 2}}
  amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %a, %b, %c
      : !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[4 : 6]>
    -> !amdgcn.vgpr<[12 : 16]>
  return
}

// -----
// Test: Wrong destination register count (2 instead of 4)

func.func @test_vop3p_mai_wrong_dst_count(%dst: !amdgcn.vgpr<[8 : 10]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.vgpr<[4 : 8]>) {
  // expected-error@+1 {{vdst operand must have 4 registers for v_mfma_f32_16x16x16_f16, got 2}}
  amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %a, %b, %c
      : !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[4 : 8]>
    -> !amdgcn.vgpr<[8 : 10]>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 DS Read Register Count Verification
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong result register count for ds_read_b128 (5 instead of 4)

func.func @test_ds_read_b128_wrong_result_count(%dst: !amdgcn.vgpr<[32 : 37]>, %addr: !amdgcn.vgpr<30>) {
  // expected-error@+1 {{dest operand must have 4 registers for ds_read_b128, got 5}}
  %tok = amdgcn.load ds_read_b128 dest %dst addr %addr : dps(!amdgcn.vgpr<[32 : 37]>) ins(!amdgcn.vgpr<30>) -> !amdgcn.read_token<shared>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 DS Write Register Count Verification
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong data register count for ds_write_b128 (3 instead of 4)

func.func @test_ds_write_b128_wrong_data_count(%addr: !amdgcn.vgpr<23>, %val0: !amdgcn.vgpr<28>, %val1: !amdgcn.vgpr<29>, %val2: !amdgcn.vgpr<30>) {
  %val_range = amdgcn.make_register_range %val0, %val1, %val2 : !amdgcn.vgpr<28>, !amdgcn.vgpr<29>, !amdgcn.vgpr<30>
  %offset = arith.constant 0 : i32
  // expected-error@+1 {{data operand must have 4 registers for ds_write_b128, got 3}}
  %tok = amdgcn.store ds_write_b128 data %val_range addr %addr : ins(!amdgcn.vgpr<[28 : 31]>, !amdgcn.vgpr<23>) -> !amdgcn.write_token<shared>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 Global Load Register Count Verification
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong result register count for global_load_dword (2 instead of 1)

func.func @test_global_load_dword_wrong_result_count(%addr_lo: !amdgcn.vgpr<40>, %addr_hi: !amdgcn.vgpr<41>, %dst: !amdgcn.vgpr<[42 : 44]>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<40>, !amdgcn.vgpr<41>
  // expected-error@+1 {{dest operand must have 1 registers for global_load_dword, got 2}}
  %tok = amdgcn.load global_load_dword dest %dst addr %addr_range : dps(!amdgcn.vgpr<[42 : 44]>) ins(!amdgcn.vgpr<[40 : 42]>) -> !amdgcn.read_token<flat>
  return
}

// -----
// Test: Wrong result register count for global_load_dwordx2 (3 instead of 2)

func.func @test_global_load_dwordx2_wrong_result_count(%addr_lo: !amdgcn.vgpr<44>, %addr_hi: !amdgcn.vgpr<45>, %dst: !amdgcn.vgpr<[44 : 47]>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<44>, !amdgcn.vgpr<45>
  // expected-error@+1 {{dest operand must have 2 registers for global_load_dwordx2, got 3}}
  %tok = amdgcn.load global_load_dwordx2 dest %dst addr %addr_range : dps(!amdgcn.vgpr<[44 : 47]>) ins(!amdgcn.vgpr<[44 : 46]>) -> !amdgcn.read_token<flat>
  return
}

// -----
// Test: Wrong result register count for global_load_dwordx3 (4 instead of 3)

func.func @test_global_load_dwordx3_wrong_result_count(%addr_lo: !amdgcn.vgpr<50>, %addr_hi: !amdgcn.vgpr<51>, %dst: !amdgcn.vgpr<[52 : 56]>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<50>, !amdgcn.vgpr<51>
  // expected-error@+1 {{dest operand must have 3 registers for global_load_dwordx3, got 4}}
  %tok = amdgcn.load global_load_dwordx3 dest %dst addr %addr_range : dps(!amdgcn.vgpr<[52 : 56]>) ins(!amdgcn.vgpr<[50 : 52]>) -> !amdgcn.read_token<flat>
  return
}

// -----
// Test: Wrong result register count for global_load_dwordx4 (5 instead of 4)

func.func @test_global_load_dwordx4_wrong_result_count(%addr_lo: !amdgcn.vgpr<58>, %addr_hi: !amdgcn.vgpr<59>, %dst: !amdgcn.vgpr<[64 : 69]>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<58>, !amdgcn.vgpr<59>
  // expected-error@+1 {{dest operand must have 4 registers for global_load_dwordx4, got 5}}
  %tok = amdgcn.load global_load_dwordx4 dest %dst addr %addr_range : dps(!amdgcn.vgpr<[64 : 69]>) ins(!amdgcn.vgpr<[58 : 60]>) -> !amdgcn.read_token<flat>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 Global Store Register Count Verification
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong addr register count for global_store_dword (2 instead of 1)

func.func @test_global_store_dword_wrong_addr_count(%addr_lo: !amdgcn.vgpr<70>, %addr_hi: !amdgcn.vgpr<71>, %val0: !amdgcn.vgpr<72>, %val1: !amdgcn.vgpr<73>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<70>, !amdgcn.vgpr<71>
  %val_range = amdgcn.make_register_range %val0, %val1 : !amdgcn.vgpr<72>, !amdgcn.vgpr<73>
  // expected-error@+1 {{data operand must have 1 registers for global_store_dword, got 2}}
  %tok = amdgcn.store global_store_dword data %val_range addr %addr_range : ins(!amdgcn.vgpr<[72 : 74]>, !amdgcn.vgpr<[70 : 72]>) -> !amdgcn.write_token<flat>
  return
}

// -----
// Test: Wrong data register count for global_store_dwordx2 (3 instead of 2)

func.func @test_global_store_dwordx2_wrong_data_count(%addr_lo: !amdgcn.vgpr<74>, %addr_hi: !amdgcn.vgpr<75>, %val0: !amdgcn.vgpr<76>, %val1: !amdgcn.vgpr<77>, %val2: !amdgcn.vgpr<78>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<74>, !amdgcn.vgpr<75>
  %val_range = amdgcn.make_register_range %val0, %val1, %val2 : !amdgcn.vgpr<76>, !amdgcn.vgpr<77>, !amdgcn.vgpr<78>
  // expected-error@+1 {{data operand must have 2 registers for global_store_dwordx2, got 3}}
  %tok = amdgcn.store global_store_dwordx2 data %val_range addr %addr_range : ins(!amdgcn.vgpr<[76 : 79]>, !amdgcn.vgpr<[74 : 76]>) -> !amdgcn.write_token<flat>
  return
}

// -----
// Test: Wrong addr register count for global_store_dwordx3 (2 instead of 3)

func.func @test_global_store_dwordx3_wrong_addr_count(%addr_lo: !amdgcn.vgpr<82>, %addr_hi: !amdgcn.vgpr<83>, %val0: !amdgcn.vgpr<84>, %val1: !amdgcn.vgpr<85>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<82>, !amdgcn.vgpr<83>
  %val_range = amdgcn.make_register_range %val0, %val1 : !amdgcn.vgpr<84>, !amdgcn.vgpr<85>
  // expected-error@+1 {{data operand must have 3 registers for global_store_dwordx3, got 2}}
  %tok = amdgcn.store global_store_dwordx3 data %val_range addr %addr_range : ins(!amdgcn.vgpr<[84 : 86]>, !amdgcn.vgpr<[82 : 84]>) -> !amdgcn.write_token<flat>
  return
}

// -----
// Test: Wrong addr register count for global_store_dwordx4 (2 instead of 4)

func.func @test_global_store_dwordx4_wrong_addr_count(%addr_lo: !amdgcn.vgpr<92>, %addr_hi: !amdgcn.vgpr<93>, %val0: !amdgcn.vgpr<96>, %val1: !amdgcn.vgpr<97>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<92>, !amdgcn.vgpr<93>
  %val_range = amdgcn.make_register_range %val0, %val1 : !amdgcn.vgpr<96>, !amdgcn.vgpr<97>
  // expected-error@+1 {{data operand must have 4 registers for global_store_dwordx4, got 2}}
  %tok = amdgcn.store global_store_dwordx4 data %val_range addr %addr_range : ins(!amdgcn.vgpr<[96 : 98]>, !amdgcn.vgpr<[92 : 94]>) -> !amdgcn.write_token<flat>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 VOP3P_MAI acc_cd Verification
//===----------------------------------------------------------------------===//

// -----
// Test: C operand must be AGPRRangeType when acc_cd is set

func.func @test_vop3p_mai_acc_cd_set_c_not_agpr(%dst: !amdgcn.agpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.vgpr<[4 : 8]>) {
  // expected-error@+1 {{all of {vdst, c} have same type IDs}}
  "amdgcn.vop3p.vop3p_mai"(%dst, %a, %b, %c) {
    opcode = #amdgcn.inst<v_mfma_f32_16x16x16_f16>
  } : (!amdgcn.agpr<[12 : 16]>, !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[4 : 8]>) -> ()
  return
}

// -----
// Test: Destination must be AGPRRangeType when acc_cd is set

func.func @test_vop3p_mai_acc_cd_set_dst_not_agpr(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.agpr<[4 : 8]>) {
  // expected-error@+1 {{all of {vdst, c} have same type IDs}}
  "amdgcn.vop3p.vop3p_mai"(%dst, %a, %b, %c) {
    opcode = #amdgcn.inst<v_mfma_f32_16x16x16_f16>
  } : (!amdgcn.vgpr<[12 : 16]>, !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.agpr<[4 : 8]>) -> ()
  return
}

// -----
// Test: C operand must be VGPRRangeType when acc_cd is not set

func.func @test_vop3p_mai_acc_cd_not_set_c_is_agpr(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.agpr<[4 : 8]>) {
  // expected-error@+1 {{all of {vdst, c} have same type IDs}}
  amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %a, %b, %c
      : !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.agpr<[4 : 8]>
    -> !amdgcn.vgpr<[12 : 16]>
  return
}

// -----
// Test: Destination must be VGPRRangeType when acc_cd is not set

func.func @test_vop3p_mai_acc_cd_not_set_dst_is_agpr(%dst: !amdgcn.agpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.vgpr<[4 : 8]>) {
  // expected-error@+1 {{all of {vdst, c} have same type IDs}}
  amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %a, %b, %c
      : !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[4 : 8]>
    -> !amdgcn.agpr<[12 : 16]>
  return
}
