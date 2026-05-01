// RUN: aster-opt %s --verify-roundtrip

func.func @sop2_add_sub_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_add_u32 - SOP2 add two unsigned 32-bit integers
  %_scc_dst_add_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %add_u32 = amdgcn.s_add_u32 outs(%sdst, %_scc_dst_add_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_sub_u32 - SOP2 subtract unsigned 32-bit integers
  %_scc_dst_sub_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %sub_u32 = amdgcn.s_sub_u32 outs(%sdst, %_scc_dst_sub_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_add_i32 - SOP2 add two signed 32-bit integers
  %_scc_dst_add_i32 = amdgcn.alloca : !amdgcn.scc<0>
  %add_i32 = amdgcn.s_add_i32 outs(%sdst, %_scc_dst_add_i32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_sub_i32 - SOP2 subtract signed 32-bit integers
  %_scc_dst_sub_i32 = amdgcn.alloca : !amdgcn.scc<0>
  %sub_i32 = amdgcn.s_sub_i32 outs(%sdst, %_scc_dst_sub_i32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return %add_u32, %sub_u32, %add_i32, %sub_i32
    :  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr
}

func.func @sop2_carry_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_addc_u32 - SOP2 add with carry-in
  %_scc_dst_addc_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %_scc_src_addc_u32 = amdgcn.alloca : !amdgcn.scc<0>
    %addc = amdgcn.s_addc_u32 outs(%sdst, %_scc_dst_addc_u32) ins(%lhs, %rhs, %_scc_src_addc_u32) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.scc<0>)
  // s_subb_u32 - SOP2 subtract with borrow
  %_scc_dst_subb_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %_scc_src_subb_u32 = amdgcn.alloca : !amdgcn.scc<0>
    %subb = amdgcn.s_subb_u32 outs(%sdst, %_scc_dst_subb_u32) ins(%lhs, %rhs, %_scc_src_subb_u32) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.scc<0>)
  return %addc, %subb :  !amdgcn.sgpr,  !amdgcn.sgpr
}

func.func @sop2_min_max_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_min_i32 - SOP2 minimum of two signed integers
  %_scc_dst_min_i32 = amdgcn.alloca : !amdgcn.scc<0>
  %min_i32 = amdgcn.s_min_i32 outs(%sdst, %_scc_dst_min_i32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_min_u32 - SOP2 minimum of two unsigned integers
  %_scc_dst_min_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %min_u32 = amdgcn.s_min_u32 outs(%sdst, %_scc_dst_min_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_max_i32 - SOP2 maximum of two signed integers
  %_scc_dst_max_i32 = amdgcn.alloca : !amdgcn.scc<0>
  %max_i32 = amdgcn.s_max_i32 outs(%sdst, %_scc_dst_max_i32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_max_u32 - SOP2 maximum of two unsigned integers
  %_scc_dst_max_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %max_u32 = amdgcn.s_max_u32 outs(%sdst, %_scc_dst_max_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return %min_i32, %min_u32, %max_i32, %max_u32
    :  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr
}


func.func @sop2_bitwise_ops(%sdst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) {
  // s_and_b32 - SOP2 bitwise AND 32-bit
  %_scc_dst_and_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %and_b32 = amdgcn.s_and_b32 outs(%sdst, %_scc_dst_and_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_or_b32 - SOP2 bitwise OR 32-bit
  %_scc_dst_or_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %or_b32 = amdgcn.s_or_b32 outs(%sdst, %_scc_dst_or_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_xor_b32 - SOP2 bitwise XOR 32-bit
  %_scc_dst_xor_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %xor_b32 = amdgcn.s_xor_b32 outs(%sdst, %_scc_dst_xor_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return %and_b32, %or_b32, %xor_b32
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
}

func.func @sop2_bitwise_negated_ops(%sdst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) {
  // s_andn2_b32 - SOP2 AND with negated second operand (32-bit)
  %_scc_dst_andn2_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %andn2_b32 = amdgcn.s_andn2_b32 outs(%sdst, %_scc_dst_andn2_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_orn2_b32 - SOP2 OR with negated second operand (32-bit)
  %_scc_dst_orn2_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %orn2_b32 = amdgcn.s_orn2_b32 outs(%sdst, %_scc_dst_orn2_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_nand_b32 - SOP2 bitwise NAND (32-bit)
  %_scc_dst_nand_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %nand_b32 = amdgcn.s_nand_b32 outs(%sdst, %_scc_dst_nand_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_nor_b32 - SOP2 bitwise NOR (32-bit)
  %_scc_dst_nor_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %nor_b32 = amdgcn.s_nor_b32 outs(%sdst, %_scc_dst_nor_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_xnor_b32 - SOP2 bitwise XNOR (32-bit)
  %_scc_dst_xnor_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %xnor_b32 = amdgcn.s_xnor_b32 outs(%sdst, %_scc_dst_xnor_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return
}

func.func @sop2_shift_ops(%sdst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) {
  // s_lshl_b32 - SOP2 logical shift left 32-bit
  %_scc_dst_lshl_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %lshl_b32 = amdgcn.s_lshl_b32 outs(%sdst, %_scc_dst_lshl_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_lshr_b32 - SOP2 logical shift right 32-bit
  %_scc_dst_lshr_b32 = amdgcn.alloca : !amdgcn.scc<0>
  %lshr_b32 = amdgcn.s_lshr_b32 outs(%sdst, %_scc_dst_lshr_b32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_ashr_i32 - SOP2 arithmetic shift right 32-bit
  %_scc_dst_ashr_i32 = amdgcn.alloca : !amdgcn.scc<0>
  %ashr_i32 = amdgcn.s_ashr_i32 outs(%sdst, %_scc_dst_ashr_i32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return
}

func.func @sop2_bitfield_ops(%sdst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) {
  // s_bfm_b32 - SOP2 create 32-bit bitfield mask
  %bfm_b32 = amdgcn.s_bfm_b32 outs(%sdst) ins(%lhs, %rhs) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_bfe_u32 - SOP2 extract unsigned bitfield from 32-bit
  %_scc_dst_bfe_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %bfe_u32 = amdgcn.s_bfe_u32 outs(%sdst, %_scc_dst_bfe_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_bfe_i32 - SOP2 extract signed bitfield from 32-bit
  %_scc_dst_bfe_i32 = amdgcn.alloca : !amdgcn.scc<0>
  %bfe_i32 = amdgcn.s_bfe_i32 outs(%sdst, %_scc_dst_bfe_i32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return
}

func.func @sop2_multiply_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_mul_i32 - SOP2 multiply two signed 32-bit integers
  %mul_i32 = amdgcn.s_mul_i32 outs(%sdst) ins(%lhs, %rhs) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_mul_hi_u32 - SOP2 multiply unsigned integers (high 32 bits)
  %mul_hi_u32 = amdgcn.s_mul_hi_u32 outs(%sdst) ins(%lhs, %rhs) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_mul_hi_i32 - SOP2 multiply signed integers (high 32 bits)
  %mul_hi_i32 = amdgcn.s_mul_hi_i32 outs(%sdst) ins(%lhs, %rhs) : outs(!amdgcn.sgpr) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return %mul_i32, %mul_hi_u32, %mul_hi_i32
    :  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr
}

func.func @sop2_special_ops(%lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr, %sdst:  !amdgcn.sgpr)
    ->  !amdgcn.sgpr {
  // s_cbranch_g_fork - SOP2 conditional branch using branch-stack (no output)

  // s_absdiff_i32 - SOP2 absolute difference of two signed integers
  %_scc_dst_absdiff_i32 = amdgcn.alloca : !amdgcn.scc<0>
  %absdiff = amdgcn.s_absdiff_i32 outs(%sdst, %_scc_dst_absdiff_i32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return %absdiff :  !amdgcn.sgpr
}

func.func @sop2_shift_add_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_lshl1_add_u32 - SOP2 shift left by 1 and add
  %_scc_dst_lshl1_add_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %lshl1_add = amdgcn.s_lshl1_add_u32 outs(%sdst, %_scc_dst_lshl1_add_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_lshl2_add_u32 - SOP2 shift left by 2 and add
  %_scc_dst_lshl2_add_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %lshl2_add = amdgcn.s_lshl2_add_u32 outs(%sdst, %_scc_dst_lshl2_add_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_lshl3_add_u32 - SOP2 shift left by 3 and add
  %_scc_dst_lshl3_add_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %lshl3_add = amdgcn.s_lshl3_add_u32 outs(%sdst, %_scc_dst_lshl3_add_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  // s_lshl4_add_u32 - SOP2 shift left by 4 and add
  %_scc_dst_lshl4_add_u32 = amdgcn.alloca : !amdgcn.scc<0>
  %lshl4_add = amdgcn.s_lshl4_add_u32 outs(%sdst, %_scc_dst_lshl4_add_u32) ins(%lhs, %rhs) : outs(!amdgcn.sgpr, !amdgcn.scc<0>) ins(!amdgcn.sgpr, !amdgcn.sgpr)
  return %lshl1_add, %lshl2_add, %lshl3_add, %lshl4_add
    :  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr
}
