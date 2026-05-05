// RUN: aster-opt %s --verify-diagnostics --split-input-file

//===----------------------------------------------------------------------===//
// CDNA3 VOP3P_MAI Register Count Verification
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong A operand register count (3 instead of 2)

func.func @test_vop3p_mai_wrong_a_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 3]>, %b: !amdgcn.vgpr<[4 : 6]>, %c: !amdgcn.vgpr<[8 : 12]>) {
  // expected-error@+1 {{'amdgcn.v_mfma_f32_16x16x16_f16' op operand #1 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {2}, sized GPR range of AGPR type with sizes {2}}}, but got '!amdgcn.vgpr<[0 : 3]>'}}
  amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[12 : 16]>)
      ins(!amdgcn.vgpr<[0 : 3]>, !amdgcn.vgpr<[4 : 6]>, !amdgcn.vgpr<[8 : 12]>)
  return
}

// -----
// Test: Wrong B operand register count (3 instead of 2)

func.func @test_vop3p_mai_wrong_b_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[4 : 7]>, %c: !amdgcn.vgpr<[8 : 12]>) {
  // expected-error@+1 {{'amdgcn.v_mfma_f32_16x16x16_f16' op operand #2 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {2}, sized GPR range of AGPR type with sizes {2}}}, but got '!amdgcn.vgpr<[4 : 7]>'}}
  amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[12 : 16]>)
      ins(!amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[4 : 7]>, !amdgcn.vgpr<[8 : 12]>)
  return
}

// -----
// Test: Wrong C operand register count (2 instead of 4)

func.func @test_vop3p_mai_wrong_c_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.vgpr<[4 : 6]>) {
  // expected-error@+1 {{'amdgcn.v_mfma_f32_16x16x16_f16' op operand #3 must be sized GPR range of VGPR type with sizes {4} or sized GPR range of AGPR type with sizes {4} or , but got '!amdgcn.vgpr<[4 : 6]>'}}
  amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[12 : 16]>)
      ins(!amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[4 : 6]>)
  return
}

// -----
// Test: Wrong destination register count (2 instead of 4)

func.func @test_vop3p_mai_wrong_dst_count(%dst: !amdgcn.vgpr<[8 : 10]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.vgpr<[4 : 8]>) {
  // expected-error@+1 {{'amdgcn.v_mfma_f32_16x16x16_f16' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {4}, sized GPR range of AGPR type with sizes {4}}}, but got '!amdgcn.vgpr<[8 : 10]>'}}
  amdgcn.v_mfma_f32_16x16x16_f16 outs(%dst) ins(%a, %b, %c)
    : outs(!amdgcn.vgpr<[8 : 10]>)
      ins(!amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[4 : 8]>)
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 DS Read Register Count Verification
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong result register count for ds_read_b128 (5 instead of 4)

func.func @test_ds_read_b128_wrong_result_count(%dst: !amdgcn.vgpr<[32 : 37]>, %addr: !amdgcn.vgpr<30>) {
  // expected-error@+2 {{'amdgcn.ds_read_b128' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {4}, sized GPR range of AGPR type with sizes {4}}}, but got '!amdgcn.vgpr<[32 : 37]>'}}
  %c0_i32_mig1 = arith.constant 0 : i32
  %tok = amdgcn.ds_read_b128 dest %dst addr %addr offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr<[32 : 37]>) ins(!amdgcn.vgpr<30>) mods(i32) -> !amdgcn.read_token<shared>
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
  // expected-error@+2 {{'amdgcn.ds_write_b128' op operand #1 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {4}, sized GPR range of AGPR type with sizes {4}}}, but got '!amdgcn.vgpr<[28 : 31]>'}}
  %c0_i32_mig1 = arith.constant 0 : i32
  %tok = amdgcn.ds_write_b128 data %val_range addr %addr offset c(%c0_i32_mig1) : ins(!amdgcn.vgpr<[28 : 31]>, !amdgcn.vgpr<23>) mods(i32) -> !amdgcn.write_token<shared>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 Global Load Register Count Verification
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong result register count for global_load_dword (2 instead of 1)

func.func @test_global_load_dword_wrong_result_count(%addr_lo: !amdgcn.vgpr<40>, %addr_hi: !amdgcn.vgpr<41>, %dst: !amdgcn.vgpr<[42 : 44]>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<40>, !amdgcn.vgpr<41>
  // expected-error@+2 {{'amdgcn.global_load_dword' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {1}, sized GPR range of AGPR type with sizes {1}}}, but got '!amdgcn.vgpr<[42 : 44]>'}}
  %c0_i32_mig1 = arith.constant 0 : i32
  %tok = amdgcn.global_load_dword dest %dst addr %addr_range offset c(%c0_i32_mig1) : outs(!amdgcn.vgpr<[42 : 44]>) ins(!amdgcn.vgpr<[40 : 42]>) mods(i32) -> !amdgcn.read_token<flat>
  return
}

// -----
// Test: Wrong result register count for global_load_dwordx2 (3 instead of 2)

func.func @test_global_load_dwordx2_wrong_result_count(%addr_lo: !amdgcn.vgpr<44>, %addr_hi: !amdgcn.vgpr<45>, %dst: !amdgcn.vgpr<[44 : 47]>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<44>, !amdgcn.vgpr<45>
  // expected-error@+2 {{'amdgcn.global_load_dwordx2' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {2}, sized GPR range of AGPR type with sizes {2}}}, but got '!amdgcn.vgpr<[44 : 47]>'}}
  %c0_i32_mig2 = arith.constant 0 : i32
  %tok = amdgcn.global_load_dwordx2 dest %dst addr %addr_range offset c(%c0_i32_mig2) : outs(!amdgcn.vgpr<[44 : 47]>) ins(!amdgcn.vgpr<[44 : 46]>) mods(i32) -> !amdgcn.read_token<flat>
  return
}

// -----
// Test: Wrong result register count for global_load_dwordx3 (4 instead of 3)

func.func @test_global_load_dwordx3_wrong_result_count(%addr_lo: !amdgcn.vgpr<50>, %addr_hi: !amdgcn.vgpr<51>, %dst: !amdgcn.vgpr<[52 : 56]>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<50>, !amdgcn.vgpr<51>
  // expected-error@+2 {{'amdgcn.global_load_dwordx3' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {3}, sized GPR range of AGPR type with sizes {3}}}, but got '!amdgcn.vgpr<[52 : 56]>'}}
  %c0_i32_mig3 = arith.constant 0 : i32
  %tok = amdgcn.global_load_dwordx3 dest %dst addr %addr_range offset c(%c0_i32_mig3) : outs(!amdgcn.vgpr<[52 : 56]>) ins(!amdgcn.vgpr<[50 : 52]>) mods(i32) -> !amdgcn.read_token<flat>
  return
}

// -----
// Test: Wrong result register count for global_load_dwordx4 (5 instead of 4)

func.func @test_global_load_dwordx4_wrong_result_count(%addr_lo: !amdgcn.vgpr<58>, %addr_hi: !amdgcn.vgpr<59>, %dst: !amdgcn.vgpr<[64 : 69]>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<58>, !amdgcn.vgpr<59>
  // expected-error@+2 {{'amdgcn.global_load_dwordx4' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {4}, sized GPR range of AGPR type with sizes {4}}}, but got '!amdgcn.vgpr<[64 : 69]>'}}
  %c0_i32_mig4 = arith.constant 0 : i32
  %tok = amdgcn.global_load_dwordx4 dest %dst addr %addr_range offset c(%c0_i32_mig4) : outs(!amdgcn.vgpr<[64 : 69]>) ins(!amdgcn.vgpr<[58 : 60]>) mods(i32) -> !amdgcn.read_token<flat>
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
  // expected-error@+2 {{'amdgcn.global_store_dword' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {1}, sized GPR range of AGPR type with sizes {1}}}, but got '!amdgcn.vgpr<[72 : 74]>'}}
  %c0_i32_mig1 = arith.constant 0 : i32
  %tok = amdgcn.global_store_dword data %val_range addr %addr_range offset c(%c0_i32_mig1) : ins(!amdgcn.vgpr<[72 : 74]>, !amdgcn.vgpr<[70 : 72]>) mods(i32) -> !amdgcn.write_token<flat>
  return
}

// -----
// Test: Wrong data register count for global_store_dwordx2 (3 instead of 2)

func.func @test_global_store_dwordx2_wrong_data_count(%addr_lo: !amdgcn.vgpr<74>, %addr_hi: !amdgcn.vgpr<75>, %val0: !amdgcn.vgpr<76>, %val1: !amdgcn.vgpr<77>, %val2: !amdgcn.vgpr<78>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<74>, !amdgcn.vgpr<75>
  %val_range = amdgcn.make_register_range %val0, %val1, %val2 : !amdgcn.vgpr<76>, !amdgcn.vgpr<77>, !amdgcn.vgpr<78>
  // expected-error@+2 {{'amdgcn.global_store_dwordx2' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {2}, sized GPR range of AGPR type with sizes {2}}}, but got '!amdgcn.vgpr<[76 : 79]>'}}
  %c0_i32_mig2 = arith.constant 0 : i32
  %tok = amdgcn.global_store_dwordx2 data %val_range addr %addr_range offset c(%c0_i32_mig2) : ins(!amdgcn.vgpr<[76 : 79]>, !amdgcn.vgpr<[74 : 76]>) mods(i32) -> !amdgcn.write_token<flat>
  return
}

// -----
// Test: Wrong addr register count for global_store_dwordx3 (2 instead of 3)

func.func @test_global_store_dwordx3_wrong_addr_count(%addr_lo: !amdgcn.vgpr<82>, %addr_hi: !amdgcn.vgpr<83>, %val0: !amdgcn.vgpr<84>, %val1: !amdgcn.vgpr<85>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<82>, !amdgcn.vgpr<83>
  %val_range = amdgcn.make_register_range %val0, %val1 : !amdgcn.vgpr<84>, !amdgcn.vgpr<85>
  // expected-error@+2 {{'amdgcn.global_store_dwordx3' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {3}, sized GPR range of AGPR type with sizes {3}}}, but got '!amdgcn.vgpr<[84 : 86]>'}}
  %c0_i32_mig3 = arith.constant 0 : i32
  %tok = amdgcn.global_store_dwordx3 data %val_range addr %addr_range offset c(%c0_i32_mig3) : ins(!amdgcn.vgpr<[84 : 86]>, !amdgcn.vgpr<[82 : 84]>) mods(i32) -> !amdgcn.write_token<flat>
  return
}

// -----
// Test: Wrong addr register count for global_store_dwordx4 (2 instead of 4)

func.func @test_global_store_dwordx4_wrong_addr_count(%addr_lo: !amdgcn.vgpr<92>, %addr_hi: !amdgcn.vgpr<93>, %val0: !amdgcn.vgpr<96>, %val1: !amdgcn.vgpr<97>) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr<92>, !amdgcn.vgpr<93>
  %val_range = amdgcn.make_register_range %val0, %val1 : !amdgcn.vgpr<96>, !amdgcn.vgpr<97>
  // expected-error@+2 {{'amdgcn.global_store_dwordx4' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {4}, sized GPR range of AGPR type with sizes {4}}}, but got '!amdgcn.vgpr<[96 : 98]>'}}
  %c0_i32_mig4 = arith.constant 0 : i32
  %tok = amdgcn.global_store_dwordx4 data %val_range addr %addr_range offset c(%c0_i32_mig4) : ins(!amdgcn.vgpr<[96 : 98]>, !amdgcn.vgpr<[92 : 94]>) mods(i32) -> !amdgcn.write_token<flat>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 VOP3P_MAI Register Count Verification (Generic Form)
//===----------------------------------------------------------------------===//

// -----
// Test: Wrong A operand register count in generic form (3 instead of 2)

func.func @test_vop3p_mai_generic_wrong_a_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 3]>, %b: !amdgcn.vgpr<[4 : 6]>, %c: !amdgcn.vgpr<[8 : 12]>) {
  // expected-error@+1 {{'amdgcn.v_mfma_f32_16x16x16_f16' op operand #1 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {2}, sized GPR range of AGPR type with sizes {2}}}, but got '!amdgcn.vgpr<[0 : 3]>'}}
  "amdgcn.v_mfma_f32_16x16x16_f16"(%dst, %a, %b, %c)
    : (!amdgcn.vgpr<[12 : 16]>, !amdgcn.vgpr<[0 : 3]>, !amdgcn.vgpr<[4 : 6]>, !amdgcn.vgpr<[8 : 12]>) -> ()
  return
}

// -----
// Test: Wrong B operand register count in generic form (3 instead of 2)

func.func @test_vop3p_mai_generic_wrong_b_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[4 : 7]>, %c: !amdgcn.vgpr<[8 : 12]>) {
  // expected-error@+1 {{'amdgcn.v_mfma_f32_16x16x16_f16' op operand #2 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {2}, sized GPR range of AGPR type with sizes {2}}}, but got '!amdgcn.vgpr<[4 : 7]>'}}
  "amdgcn.v_mfma_f32_16x16x16_f16"(%dst, %a, %b, %c)
    : (!amdgcn.vgpr<[12 : 16]>, !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[4 : 7]>, !amdgcn.vgpr<[8 : 12]>) -> ()
  return
}

// -----
// Test: Wrong C operand register count in generic form (2 instead of 4)

func.func @test_vop3p_mai_generic_wrong_c_count(%dst: !amdgcn.vgpr<[12 : 16]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.vgpr<[4 : 6]>) {
  // expected-error@+1 {{'amdgcn.v_mfma_f32_16x16x16_f16' op operand #3 must be sized GPR range of VGPR type with sizes {4} or sized GPR range of AGPR type with sizes {4} or , but got '!amdgcn.vgpr<[4 : 6]>'}}
  "amdgcn.v_mfma_f32_16x16x16_f16"(%dst, %a, %b, %c)
    : (!amdgcn.vgpr<[12 : 16]>, !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[4 : 6]>) -> ()
  return
}

// -----
// Test: Wrong destination register count in generic form (2 instead of 4)

func.func @test_vop3p_mai_generic_wrong_dst_count(%dst: !amdgcn.vgpr<[12 : 14]>, %a: !amdgcn.vgpr<[0 : 2]>, %b: !amdgcn.vgpr<[2 : 4]>, %c: !amdgcn.vgpr<[4 : 8]>) {
  // expected-error@+1 {{'amdgcn.v_mfma_f32_16x16x16_f16' op operand #0 must be register type of: {register type of: {sized GPR range of VGPR type with sizes {4}, sized GPR range of AGPR type with sizes {4}}}, but got '!amdgcn.vgpr<[12 : 14]>'}}
  "amdgcn.v_mfma_f32_16x16x16_f16"(%dst, %a, %b, %c)
    : (!amdgcn.vgpr<[12 : 14]>, !amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<[2 : 4]>, !amdgcn.vgpr<[4 : 8]>) -> ()
  return
}
