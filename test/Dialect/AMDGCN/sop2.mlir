// RUN: aster-opt %s --verify-roundtrip

func.func @sop2_add_sub_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_add_u32 - SOP2 add two unsigned 32-bit integers
  %add_u32 = amdgcn.sop2 s_add_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_sub_u32 - SOP2 subtract unsigned 32-bit integers
  %sub_u32 = amdgcn.sop2 s_sub_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_add_i32 - SOP2 add two signed 32-bit integers
  %add_i32 = amdgcn.sop2 s_add_i32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_sub_i32 - SOP2 subtract signed 32-bit integers
  %sub_i32 = amdgcn.sop2 s_sub_i32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return %add_u32, %sub_u32, %add_i32, %sub_i32
    :  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr
}

func.func @sop2_carry_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_addc_u32 - SOP2 add with carry-in
  %addc = amdgcn.sop2 s_addc_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_subb_u32 - SOP2 subtract with borrow
  %subb = amdgcn.sop2 s_subb_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return %addc, %subb :  !amdgcn.sgpr,  !amdgcn.sgpr
}

func.func @sop2_min_max_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_min_i32 - SOP2 minimum of two signed integers
  %min_i32 = amdgcn.sop2 s_min_i32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_min_u32 - SOP2 minimum of two unsigned integers
  %min_u32 = amdgcn.sop2 s_min_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_max_i32 - SOP2 maximum of two signed integers
  %max_i32 = amdgcn.sop2 s_max_i32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_max_u32 - SOP2 maximum of two unsigned integers
  %max_u32 = amdgcn.sop2 s_max_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return %min_i32, %min_u32, %max_i32, %max_u32
    :  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr
}


func.func @sop2_bitwise_ops(%sdst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) {
  // s_and_b32 - SOP2 bitwise AND 32-bit
  %and_b32 = amdgcn.sop2 s_and_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_or_b32 - SOP2 bitwise OR 32-bit
  %or_b32 = amdgcn.sop2 s_or_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_xor_b32 - SOP2 bitwise XOR 32-bit
  %xor_b32 = amdgcn.sop2 s_xor_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return %and_b32, %or_b32, %xor_b32
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
}

func.func @sop2_bitwise_negated_ops(%sdst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) {
  // s_andn2_b32 - SOP2 AND with negated second operand (32-bit)
  %andn2_b32 = amdgcn.sop2 s_andn2_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_orn2_b32 - SOP2 OR with negated second operand (32-bit)
  %orn2_b32 = amdgcn.sop2 s_orn2_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_nand_b32 - SOP2 bitwise NAND (32-bit)
  %nand_b32 = amdgcn.sop2 s_nand_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_nor_b32 - SOP2 bitwise NOR (32-bit)
  %nor_b32 = amdgcn.sop2 s_nor_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  // s_xnor_b32 - SOP2 bitwise XNOR (32-bit)
  %xnor_b32 = amdgcn.sop2 s_xnor_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return
}

func.func @sop2_shift_ops(%sdst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) {
  // s_lshl_b32 - SOP2 logical shift left 32-bit
  %lshl_b32 = amdgcn.sop2 s_lshl_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_lshr_b32 - SOP2 logical shift right 32-bit
  %lshr_b32 = amdgcn.sop2 s_lshr_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_ashr_i32 - SOP2 arithmetic shift right 32-bit
  %ashr_i32 = amdgcn.sop2 s_ashr_i32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return
}

func.func @sop2_bitfield_ops(%sdst: !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr) {
  // s_bfm_b32 - SOP2 create 32-bit bitfield mask
  %bfm_b32 = amdgcn.sop2 s_bfm_b32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_bfe_u32 - SOP2 extract unsigned bitfield from 32-bit
  %bfe_u32 = amdgcn.sop2 s_bfe_u32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_bfe_i32 - SOP2 extract signed bitfield from 32-bit
  %bfe_i32 = amdgcn.sop2 s_bfe_i32 outs %sdst ins %lhs, %rhs
    : !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return
}

func.func @sop2_multiply_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_mul_i32 - SOP2 multiply two signed 32-bit integers
  %mul_i32 = amdgcn.sop2 s_mul_i32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_mul_hi_u32 - SOP2 multiply unsigned integers (high 32 bits)
  %mul_hi_u32 = amdgcn.sop2 s_mul_hi_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_mul_hi_i32 - SOP2 multiply signed integers (high 32 bits)
  %mul_hi_i32 = amdgcn.sop2 s_mul_hi_i32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return %mul_i32, %mul_hi_u32, %mul_hi_i32
    :  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr
}

func.func @sop2_special_ops(%lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr, %sdst:  !amdgcn.sgpr)
    ->  !amdgcn.sgpr {
  // s_cbranch_g_fork - SOP2 conditional branch using branch-stack (no output)
  amdgcn.sop2 s_cbranch_g_fork ins %lhs, %rhs : !amdgcn.sgpr, !amdgcn.sgpr

  // s_absdiff_i32 - SOP2 absolute difference of two signed integers
  %absdiff = amdgcn.sop2 s_absdiff_i32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return %absdiff :  !amdgcn.sgpr
}

func.func @sop2_shift_add_ops(%sdst:  !amdgcn.sgpr, %lhs: !amdgcn.sgpr, %rhs: !amdgcn.sgpr)
    -> ( !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr) {
  // s_lshl1_add_u32 - SOP2 shift left by 1 and add
  %lshl1_add = amdgcn.sop2 s_lshl1_add_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_lshl2_add_u32 - SOP2 shift left by 2 and add
  %lshl2_add = amdgcn.sop2 s_lshl2_add_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_lshl3_add_u32 - SOP2 shift left by 3 and add
  %lshl3_add = amdgcn.sop2 s_lshl3_add_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  // s_lshl4_add_u32 - SOP2 shift left by 4 and add
  %lshl4_add = amdgcn.sop2 s_lshl4_add_u32 outs %sdst ins %lhs, %rhs
    :  !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr

  return %lshl1_add, %lshl2_add, %lshl3_add, %lshl4_add
    :  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr,  !amdgcn.sgpr
}
