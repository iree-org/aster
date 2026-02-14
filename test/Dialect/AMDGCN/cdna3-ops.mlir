// RUN: aster-opt %s --verify-roundtrip

//===----------------------------------------------------------------------===//
// CDNA3 VOP3P_MAI Operations
//===----------------------------------------------------------------------===//

func.func @test_vop3p_mai_basic(%a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>, %c: !amdgcn.vgpr<[? + 4]>, %dst0: !amdgcn.vgpr<[? + 4]>, %dst1: !amdgcn.vgpr<[? + 4]>) {
  %result = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst0, %a, %b, %c
      : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  %result2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_bf16> %dst1, %a, %b, %result
      : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>
  return
}

func.func @test_vop3p_mai_with_agprs(%a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>, %c: !amdgcn.agpr<[? + 4]>, %dst: !amdgcn.agpr<[? + 4]>) {
  %result = "amdgcn.vop3p.vop3p_mai"(%dst, %a, %b, %c) {
    opcode = #amdgcn.inst<v_mfma_f32_16x16x16_f16>,
    acc_cd
  } : (!amdgcn.agpr<[? + 4]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.agpr<[? + 4]>) -> !amdgcn.agpr<[? + 4]>
  return
}

func.func @test_vop3p_mai_oilist_attributes(%a: !amdgcn.vgpr<[? + 2]>, %b: !amdgcn.vgpr<[? + 2]>, %c: !amdgcn.vgpr<[? + 4]>, %dst: !amdgcn.vgpr<[? + 4]>, %c_a: !amdgcn.agpr<[? + 4]>, %dst_a: !amdgcn.agpr<[? + 4]>) {
  // Test that acc_cd can appear without cbsz (oilist allows any order)
  %result1 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst_a, %a, %b, %c_a
      : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.agpr<[? + 4]>
    -> !amdgcn.agpr<[? + 4]>

  // Test attributes in different order: blgp before cbsz
  %result2 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst, %a, %b, %c blgp = 1 cbsz = 2
      : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 4]>
    -> !amdgcn.vgpr<[? + 4]>

  // Test all attributes in reverse order
  %result3 = amdgcn.vop3p.vop3p_mai #amdgcn.inst<v_mfma_f32_16x16x16_f16> %dst_a, %a, %b, %c_a blgp = 3 abid = 4 cbsz = 0
      : !amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>, !amdgcn.agpr<[? + 4]>
    -> !amdgcn.agpr<[? + 4]>

  return
}

//===----------------------------------------------------------------------===//
// CDNA3 DS Read Operations
//===----------------------------------------------------------------------===//

func.func @test_ds_read_b32(%addr: !amdgcn.vgpr, %d: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %offset = arith.constant 0 : i32
  %result, %tok = amdgcn.load ds_read_b32 dest %d addr %addr offset c(%offset) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
  %0 = amdgcn.split_register_range %result : !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_ds_read_b32_with_offset(%addr: !amdgcn.vgpr, %dst1: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %offset = arith.constant 4 : i32
  %result, %tok = amdgcn.load ds_read_b32 dest %dst1 addr %addr offset c(%offset) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
  %0 = amdgcn.split_register_range %result : !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_ds_read_b64(%addr: !amdgcn.vgpr, %dst2: !amdgcn.vgpr<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %offset = arith.constant 0 : i32
  %result, %tok = amdgcn.load ds_read_b64 dest %dst2 addr %addr offset c(%offset) : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
  %0, %1 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 2]>
  return %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
}

func.func @test_ds_read_b96(%addr: !amdgcn.vgpr, %dst3: !amdgcn.vgpr<[? + 3]>) -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) {
  %offset = arith.constant 0 : i32
  %result, %tok = amdgcn.load ds_read_b96 dest %dst3 addr %addr offset c(%offset) : dps(!amdgcn.vgpr<[? + 3]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
  %0, %1, %2 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 3]>
  return %0, %1, %2 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
}

func.func @test_ds_read_b128(%addr: !amdgcn.vgpr, %dst4: !amdgcn.vgpr<[? + 4]>) -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) {
  %offset = arith.constant 0 : i32
  %result, %tok = amdgcn.load ds_read_b128 dest %dst4 addr %addr offset c(%offset) : dps(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
  %0, %1, %2, %3 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 4]>
  return %0, %1, %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
}

//===----------------------------------------------------------------------===//
// CDNA3 DS Write Operations
//===----------------------------------------------------------------------===//

func.func @test_ds_write_b32(%addr: !amdgcn.vgpr, %val: !amdgcn.vgpr) {
  %val_range = amdgcn.make_register_range %val : !amdgcn.vgpr
  %offset = arith.constant 0 : i32
  %tok = amdgcn.store ds_write_b32 data %val_range addr %addr offset c(%offset) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

func.func @test_ds_write_b32_with_offset(%addr: !amdgcn.vgpr, %val: !amdgcn.vgpr) {
  %val_range = amdgcn.make_register_range %val : !amdgcn.vgpr
  %offset = arith.constant 8 : i32
  %tok = amdgcn.store ds_write_b32 data %val_range addr %addr offset c(%offset) : ins(!amdgcn.vgpr, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

func.func @test_ds_write_b64(%addr: !amdgcn.vgpr, %val0: !amdgcn.vgpr, %val1: !amdgcn.vgpr) {
  %val_range = amdgcn.make_register_range %val0, %val1 : !amdgcn.vgpr, !amdgcn.vgpr
  %offset = arith.constant 0 : i32
  %tok = amdgcn.store ds_write_b64 data %val_range addr %addr offset c(%offset) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

func.func @test_ds_write_b64_with_offset(%addr: !amdgcn.vgpr, %val0: !amdgcn.vgpr, %val1: !amdgcn.vgpr) {
  %val_range = amdgcn.make_register_range %val0, %val1 : !amdgcn.vgpr, !amdgcn.vgpr
  %offset = arith.constant 16 : i32
  %tok = amdgcn.store ds_write_b64 data %val_range addr %addr offset c(%offset) : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

func.func @test_ds_write_b96(%addr: !amdgcn.vgpr, %val0: !amdgcn.vgpr, %val1: !amdgcn.vgpr, %val2: !amdgcn.vgpr) {
  %val_range = amdgcn.make_register_range %val0, %val1, %val2 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  %offset = arith.constant 0 : i32
  %tok = amdgcn.store ds_write_b96 data %val_range addr %addr offset c(%offset) : ins(!amdgcn.vgpr<[? + 3]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

func.func @test_ds_write_b128(%addr: !amdgcn.vgpr, %val0: !amdgcn.vgpr, %val1: !amdgcn.vgpr, %val2: !amdgcn.vgpr, %val3: !amdgcn.vgpr) {
  %val_range = amdgcn.make_register_range %val0, %val1, %val2, %val3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  %offset = arith.constant 0 : i32
  %tok = amdgcn.store ds_write_b128 data %val_range addr %addr offset c(%offset) : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr, i32) -> !amdgcn.write_token<shared>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 Global Load Operations
//===----------------------------------------------------------------------===//

func.func @test_global_load_dword(%addr_lo: !amdgcn.vgpr, %addr_hi: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %result, %tok = amdgcn.load global_load_dword dest %dst addr %addr_range : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %0 = amdgcn.split_register_range %result : !amdgcn.vgpr
  return %0 : !amdgcn.vgpr
}

func.func @test_global_load_dwordx2(%addr_lo: !amdgcn.vgpr, %addr_hi: !amdgcn.vgpr, %dst: !amdgcn.vgpr<[? + 2]>) -> (!amdgcn.vgpr, !amdgcn.vgpr) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %result, %tok = amdgcn.load global_load_dwordx2 dest %dst addr %addr_range : dps(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %0, %1 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 2]>
  return %0, %1 : !amdgcn.vgpr, !amdgcn.vgpr
}

func.func @test_global_load_dwordx3(%addr_lo: !amdgcn.vgpr, %addr_hi: !amdgcn.vgpr, %dst: !amdgcn.vgpr<[? + 3]>) -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %result, %tok = amdgcn.load global_load_dwordx3 dest %dst addr %addr_range : dps(!amdgcn.vgpr<[? + 3]>) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %0, %1, %2 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 3]>
  return %0, %1, %2 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
}

func.func @test_global_load_dwordx4(%addr_lo: !amdgcn.vgpr, %addr_hi: !amdgcn.vgpr, %dst: !amdgcn.vgpr<[? + 4]>) -> (!amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %result, %tok = amdgcn.load global_load_dwordx4 dest %dst addr %addr_range : dps(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr<[? + 2]>) -> !amdgcn.read_token<flat>
  %0, %1, %2, %3 = amdgcn.split_register_range %result : !amdgcn.vgpr<[? + 4]>
  return %0, %1, %2, %3 : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
}

//===----------------------------------------------------------------------===//
// CDNA3 Global Store Operations
//===----------------------------------------------------------------------===//

func.func @test_global_store_dword(%addr_lo: !amdgcn.vgpr, %addr_hi: !amdgcn.vgpr, %val: !amdgcn.vgpr) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %val_range = amdgcn.make_register_range %val : !amdgcn.vgpr
  %tok = amdgcn.store global_store_dword data %val_range addr %addr_range : ins(!amdgcn.vgpr, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
  return
}

func.func @test_global_store_dwordx2(%addr_lo: !amdgcn.vgpr, %addr_hi: !amdgcn.vgpr, %val_lo: !amdgcn.vgpr, %val_hi: !amdgcn.vgpr) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %val_range = amdgcn.make_register_range %val_lo, %val_hi : !amdgcn.vgpr, !amdgcn.vgpr
  %tok = amdgcn.store global_store_dwordx2 data %val_range addr %addr_range : ins(!amdgcn.vgpr<[? + 2]>, !amdgcn.vgpr<[? + 2]>) -> !amdgcn.write_token<flat>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 SMEM Load Operations
//===----------------------------------------------------------------------===//

func.func @test_smem_load_dword(%addr_lo: !amdgcn.sgpr, %addr_hi: !amdgcn.sgpr, %dst: !amdgcn.sgpr) -> !amdgcn.sgpr {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.sgpr, !amdgcn.sgpr
  %result, %tok = amdgcn.load s_load_dword dest %dst addr %addr_range : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>) -> !amdgcn.read_token<constant>
  %0 = amdgcn.split_register_range %result : !amdgcn.sgpr
  return %0 : !amdgcn.sgpr
}

//===----------------------------------------------------------------------===//
// CDNA3 SMEM Store Operations
//===----------------------------------------------------------------------===//

func.func @test_smem_store_dword(%addr_lo: !amdgcn.sgpr, %addr_hi: !amdgcn.sgpr, %val: !amdgcn.sgpr) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.sgpr, !amdgcn.sgpr
  %val_range = amdgcn.make_register_range %val : !amdgcn.sgpr
  %tok = amdgcn.store s_store_dword data %val_range addr %addr_range : ins(!amdgcn.sgpr, !amdgcn.sgpr<[? + 2]>) -> !amdgcn.write_token<constant>
  return
}

func.func @test_smem_store_dwordx4(%addr_lo: !amdgcn.sgpr, %addr_hi: !amdgcn.sgpr, %val0: !amdgcn.sgpr, %val1: !amdgcn.sgpr, %val2: !amdgcn.sgpr, %val3: !amdgcn.sgpr) {
  %addr_range = amdgcn.make_register_range %addr_lo, %addr_hi : !amdgcn.sgpr, !amdgcn.sgpr
  %val_range = amdgcn.make_register_range %val0, %val1, %val2, %val3 : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  %tok = amdgcn.store s_store_dwordx4 data %val_range addr %addr_range : ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>) -> !amdgcn.write_token<constant>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 SOPP Operations
//===----------------------------------------------------------------------===//

func.func @test_sopp_waitcnt() {
  amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0 expcnt = 0 lgkmcnt = 0
  return
}

func.func @test_sopp_waitcnt_with_imm() {
  amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 5 expcnt = 2 lgkmcnt = 1
  return
}

func.func @test_sopp_trap() {
  amdgcn.sopp.sopp #amdgcn.inst<s_trap>
  return
}

func.func @test_sopp_trap_with_imm() {
  amdgcn.sopp.sopp #amdgcn.inst<s_trap> , imm = 2
  return
}

func.func @test_sopp_barrier() {
  amdgcn.sopp.sopp #amdgcn.inst<s_barrier>
  return
}

//===----------------------------------------------------------------------===//
// CDNA3 VOP2 Operations
//===----------------------------------------------------------------------===//

func.func @test_vop2_lshrrev_b32_e32_vgpr(%src0: !amdgcn.vgpr, %vsrc1: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop2 v_lshrrev_b32_e32 outs %dst ins %src0, %vsrc1
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop2_lshrrev_b32_e32_sgpr(%src0: !amdgcn.sgpr, %vsrc1: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop2 v_lshrrev_b32_e32 outs %dst ins %src0, %vsrc1
      : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop2_lshrrev_b32_e32_imm(%vsrc1: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c8 = arith.constant 8 : i32
  %result = amdgcn.vop2 v_lshrrev_b32_e32 outs %dst ins %c8, %vsrc1
      : !amdgcn.vgpr, i32, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop2_lshlrev_b32_e32_vgpr(%src0: !amdgcn.vgpr, %vsrc1: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop2 v_lshlrev_b32_e32 outs %dst ins %src0, %vsrc1
      : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop2_lshlrev_b32_e32_sgpr(%src0: !amdgcn.sgpr, %vsrc1: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop2 v_lshlrev_b32_e32 outs %dst ins %src0, %vsrc1
      : !amdgcn.vgpr, !amdgcn.sgpr, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop2_lshlrev_b32_e32_imm(%vsrc1: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c8 = arith.constant 8 : i32
  %result = amdgcn.vop2 v_lshlrev_b32_e32 outs %dst ins %c8, %vsrc1
      : !amdgcn.vgpr, i32, !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

//===----------------------------------------------------------------------===//
// CDNA3 VOP1 Operations
//===----------------------------------------------------------------------===//

func.func @test_vop1_mov_b32_e32_vgpr(%src0: !amdgcn.vgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop1_mov_b32_e32_sgpr(%src0: !amdgcn.sgpr, %dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %dst, %src0
      : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}

func.func @test_vop1_mov_b32_e32_imm(%dst: !amdgcn.vgpr) -> !amdgcn.vgpr {
  %c42 = arith.constant 42 : i32
  %result = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %dst, %c42
      : (!amdgcn.vgpr, i32) -> !amdgcn.vgpr
  return %result : !amdgcn.vgpr
}
