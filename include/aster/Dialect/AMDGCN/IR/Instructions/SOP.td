//===- SOP.td - AMDGCN SOP Instructions --------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_SOP_TD
#define AMDGCN_SOP_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

#ifndef SOP2OUT2IN2INST_DEF
#define SOP2OUT2IN2INST_DEF
class SOP2Out2In2Inst<string mnemonic, list<Type> dst0Type, list<Type> src0Type, list<Type> src1Type> : AMDISAInstruction<mnemonic> {
  let props = [IsSALU, IsSOP2];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dst0,
    SCCType:$scc_dst
  );
  let inputs = (ins
    AnyOperandOf<src0Type>:$src0,
    AnyOperandOf<src1Type>:$src1
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], "${mnemonic} $dst0, $src0, $src1">
  ];
}
#endif // SOP2OUT2IN2INST_DEF

#ifndef SABSDIFFI32_DEF
#define SABSDIFFI32_DEF
def SAbsdiffI32 : SOP2Out2In2Inst<"s_absdiff_i32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate the absolute value of difference between two scalar inputs,
    store the result into a scalar register and set SCC iff the result is
    nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0101010>];
}
#endif // SABSDIFFI32_DEF

#ifndef SADDI32_DEF
#define SADDI32_DEF
def SAddI32 : SOP2Out2In2Inst<"s_add_i32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Add two signed 32-bit integer inputs, store the result into a scalar
    register and store the carry-out bit into SCC.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0000010>];
}
#endif // SADDI32_DEF

#ifndef SADDU32_DEF
#define SADDU32_DEF
def SAddU32 : SOP2Out2In2Inst<"s_add_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Add two unsigned 32-bit integer inputs, store the result into a scalar
    register and store the carry-out bit into SCC.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0000000>];
}
#endif // SADDU32_DEF

#ifndef SANDN2B32_DEF
#define SANDN2B32_DEF
def SAndn2B32 : SOP2Out2In2Inst<"s_andn2_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate bitwise AND with the first input and the negation of the second
    input, store the result into a scalar register and set SCC if the result
    is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0010010>];
}
#endif // SANDN2B32_DEF

#ifndef SANDB32_DEF
#define SANDB32_DEF
def SAndB32 : SOP2Out2In2Inst<"s_and_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate bitwise AND on two scalar inputs, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0001100>];
}
#endif // SANDB32_DEF

#ifndef SANDB64_DEF
#define SANDB64_DEF
def SAndB64 : SOP2Out2In2Inst<"s_and_b64",
    /*dst0=*/[TwoSGPROrSRegType],
    /*src0=*/[TwoSGPROrSRegOrIntConstType],
    /*src1=*/[TwoSGPROrSRegOrIntConstType]
  > {
  let description = [{
    Calculate bitwise AND on two scalar inputs, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0001101>];
}
#endif // SANDB64_DEF

#ifndef SASHRI32_DEF
#define SASHRI32_DEF
def SAshrI32 : SOP2Out2In2Inst<"s_ashr_i32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Given a shift count in the second scalar input, calculate the arithmetic
    shift right (preserving sign bit) of the first scalar input, store the
    result into a scalar register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0100000>];
}
#endif // SASHRI32_DEF

#ifndef SASHRI64_DEF
#define SASHRI64_DEF
def SAshrI64 : SOP2Out2In2Inst<"s_ashr_i64",
    /*dst0=*/[TwoSGPROrSRegType],
    /*src0=*/[TwoSGPROrSRegOrIntConstType],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Given a shift count in the second scalar input, calculate the arithmetic
    shift right (preserving sign bit) of the first scalar input, store the
    result into a scalar register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0100001>];
}
#endif // SASHRI64_DEF

#ifndef SBFEI32_DEF
#define SBFEI32_DEF
def SBfeI32 : SOP2Out2In2Inst<"s_bfe_i32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Extract a signed bitfield from the first input using field offset and size
    encoded in the second input, store the result into a scalar register and
    set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0100110>];
}
#endif // SBFEI32_DEF

#ifndef SBFEU32_DEF
#define SBFEU32_DEF
def SBfeU32 : SOP2Out2In2Inst<"s_bfe_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Extract an unsigned bitfield from the first input using field offset and
    size encoded in the second input, store the result into a scalar register
    and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0100101>];
}
#endif // SBFEU32_DEF

#ifndef SLSHL1ADDU32_DEF
#define SLSHL1ADDU32_DEF
def SLshl1AddU32 : SOP2Out2In2Inst<"s_lshl1_add_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate the logical shift left of the first input by 1, then add the
    second input, store the result into a scalar register and set SCC iff the
    summation results in an unsigned overflow.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0101110>];
}
#endif // SLSHL1ADDU32_DEF

#ifndef SLSHL2ADDU32_DEF
#define SLSHL2ADDU32_DEF
def SLshl2AddU32 : SOP2Out2In2Inst<"s_lshl2_add_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate the logical shift left of the first input by 2, then add the
    second input, store the result into a scalar register and set SCC iff the
    summation results in an unsigned overflow.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0101111>];
}
#endif // SLSHL2ADDU32_DEF

#ifndef SLSHL3ADDU32_DEF
#define SLSHL3ADDU32_DEF
def SLshl3AddU32 : SOP2Out2In2Inst<"s_lshl3_add_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate the logical shift left of the first input by 3, then add the
    second input, store the result into a scalar register and set SCC iff the
    summation results in an unsigned overflow.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0110000>];
}
#endif // SLSHL3ADDU32_DEF

#ifndef SLSHL4ADDU32_DEF
#define SLSHL4ADDU32_DEF
def SLshl4AddU32 : SOP2Out2In2Inst<"s_lshl4_add_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate the logical shift left of the first input by 4, then add the
    second input, store the result into a scalar register and set SCC iff the
    summation results in an unsigned overflow.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0110001>];
}
#endif // SLSHL4ADDU32_DEF

#ifndef SLSHLB32_DEF
#define SLSHLB32_DEF
def SLshlB32 : SOP2Out2In2Inst<"s_lshl_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Given a shift count in the second scalar input, calculate the logical
    shift left of the first scalar input, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0011100>];
}
#endif // SLSHLB32_DEF

#ifndef SLSHLB64_DEF
#define SLSHLB64_DEF
def SLshlB64 : SOP2Out2In2Inst<"s_lshl_b64",
    /*dst0=*/[TwoSGPROrSRegType],
    /*src0=*/[TwoSGPROrSRegOrIntConstType],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Given a shift count in the second scalar input, calculate the logical
    shift left of the first scalar input, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0011101>];
}
#endif // SLSHLB64_DEF

#ifndef SLSHRB32_DEF
#define SLSHRB32_DEF
def SLshrB32 : SOP2Out2In2Inst<"s_lshr_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Given a shift count in the second scalar input, calculate the logical
    shift right of the first scalar input, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0011110>];
}
#endif // SLSHRB32_DEF

#ifndef SLSHRB64_DEF
#define SLSHRB64_DEF
def SLshrB64 : SOP2Out2In2Inst<"s_lshr_b64",
    /*dst0=*/[TwoSGPROrSRegType],
    /*src0=*/[TwoSGPROrSRegOrIntConstType],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Given a shift count in the second scalar input, calculate the logical
    shift right of the first scalar input, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0011111>];
}
#endif // SLSHRB64_DEF

#ifndef SMAXI32_DEF
#define SMAXI32_DEF
def SMaxI32 : SOP2Out2In2Inst<"s_max_i32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Select the maximum of two signed 32-bit integer inputs, store the selected
    value into a scalar register and set SCC iff the first value is selected.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0001000>];
}
#endif // SMAXI32_DEF

#ifndef SMAXU32_DEF
#define SMAXU32_DEF
def SMaxU32 : SOP2Out2In2Inst<"s_max_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Select the maximum of two unsigned 32-bit integer inputs, store the
    selected value into a scalar register and set SCC iff the first value is
    selected.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0001001>];
}
#endif // SMAXU32_DEF

#ifndef SMINI32_DEF
#define SMINI32_DEF
def SMinI32 : SOP2Out2In2Inst<"s_min_i32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Select the minimum of two signed 32-bit integer inputs, store the selected
    value into a scalar register and set SCC iff the first value is selected.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0000110>];
}
#endif // SMINI32_DEF

#ifndef SMINU32_DEF
#define SMINU32_DEF
def SMinU32 : SOP2Out2In2Inst<"s_min_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Select the minimum of two unsigned 32-bit integer inputs, store the
    selected value into a scalar register and set SCC iff the first value is
    selected.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0000111>];
}
#endif // SMINU32_DEF

#ifndef SNANDB32_DEF
#define SNANDB32_DEF
def SNandB32 : SOP2Out2In2Inst<"s_nand_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate bitwise NAND on two scalar inputs, store the result into a
    scalar register and set SCC if the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0010110>];
}
#endif // SNANDB32_DEF

#ifndef SNORB32_DEF
#define SNORB32_DEF
def SNorB32 : SOP2Out2In2Inst<"s_nor_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate bitwise NOR on two scalar inputs, store the result into a scalar
    register and set SCC if the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0011000>];
}
#endif // SNORB32_DEF

#ifndef SORN2B32_DEF
#define SORN2B32_DEF
def SOrn2B32 : SOP2Out2In2Inst<"s_orn2_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate bitwise OR with the first input and the negation of the second
    input, store the result into a scalar register and set SCC if the result
    is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0010100>];
}
#endif // SORN2B32_DEF

#ifndef SORB32_DEF
#define SORB32_DEF
def SOrB32 : SOP2Out2In2Inst<"s_or_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate bitwise OR on two scalar inputs, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0001110>];
}
#endif // SORB32_DEF

#ifndef SORB64_DEF
#define SORB64_DEF
def SOrB64 : SOP2Out2In2Inst<"s_or_b64",
    /*dst0=*/[TwoSGPROrSRegType],
    /*src0=*/[TwoSGPROrSRegOrIntConstType],
    /*src1=*/[TwoSGPROrSRegOrIntConstType]
  > {
  let description = [{
    Calculate bitwise OR on two scalar inputs, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0001111>];
}
#endif // SORB64_DEF

#ifndef SSUBI32_DEF
#define SSUBI32_DEF
def SSubI32 : SOP2Out2In2Inst<"s_sub_i32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Subtract the second signed 32-bit integer input from the first input,
    store the result into a scalar register and store the carry-out bit into
    SCC.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0000011>];
}
#endif // SSUBI32_DEF

#ifndef SSUBU32_DEF
#define SSUBU32_DEF
def SSubU32 : SOP2Out2In2Inst<"s_sub_u32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Subtract the second unsigned 32-bit integer input from the first input,
    store the result into a scalar register and store the carry-out bit into
    SCC.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0000001>];
}
#endif // SSUBU32_DEF

#ifndef SXNORB32_DEF
#define SXNORB32_DEF
def SXnorB32 : SOP2Out2In2Inst<"s_xnor_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate bitwise XNOR on two scalar inputs, store the result into a
    scalar register and set SCC if the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0011010>];
}
#endif // SXNORB32_DEF

#ifndef SXORB32_DEF
#define SXORB32_DEF
def SXorB32 : SOP2Out2In2Inst<"s_xor_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]
  > {
  let description = [{
    Calculate bitwise XOR on two scalar inputs, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0010000>];
}
#endif // SXORB32_DEF

#ifndef SXORB64_DEF
#define SXORB64_DEF
def SXorB64 : SOP2Out2In2Inst<"s_xor_b64",
    /*dst0=*/[TwoSGPROrSRegType],
    /*src0=*/[TwoSGPROrSRegOrIntConstType],
    /*src1=*/[TwoSGPROrSRegOrIntConstType]
  > {
  let description = [{
    Calculate bitwise XOR on two scalar inputs, store the result into a scalar
    register and set SCC iff the result is nonzero.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0010001>];
}
#endif // SXORB64_DEF

#ifndef SOP2OUT2IN3INST_DEF
#define SOP2OUT2IN3INST_DEF
class SOP2Out2In3Inst<string mnemonic> : AMDISAInstruction<mnemonic> {
  let props = [IsSALU, IsSOP2];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    OneSGPROrSRegType:$dst0,
    SCCType:$scc_dst
  );
  let inputs = (ins
    OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
    OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src1,
    SCCType:$scc_src
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], "${mnemonic} $dst0, $src0, $src1">
  ];
}
#endif // SOP2OUT2IN3INST_DEF

#ifndef SADDCU32_DEF
#define SADDCU32_DEF
def SAddcU32 : SOP2Out2In3Inst<"s_addc_u32"> {
  let description = [{
    Add two unsigned 32-bit integer inputs and a carry-in bit from SCC, store
    the result into a scalar register and store the carry-out bit into SCC.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0000100>];
}
#endif // SADDCU32_DEF

#ifndef SSUBBU32_DEF
#define SSUBBU32_DEF
def SSubbU32 : SOP2Out2In3Inst<"s_subb_u32"> {
  let description = [{
    Subtract the second unsigned 32-bit integer input from the first input,
    subtract the carry-in bit, store the result into a scalar register and
    store the carry-out bit into SCC.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0000101>];
}
#endif // SSUBBU32_DEF

#ifndef SBARRIER_DEF
#define SBARRIER_DEF
def SBarrier : AMDISAInstruction<"s_barrier"> {
  let props = [IsSALU, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Synchronize waves within a threadgroup.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0001010>];
  let outputs = (ins);
  let inputs = (ins);
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic}">
  ];
}
#endif // SBARRIER_DEF

#ifndef SOP2OUT1IN2INST_DEF
#define SOP2OUT1IN2INST_DEF
class SOP2Out1In2Inst<string mnemonic> : AMDISAInstruction<mnemonic> {
  let props = [IsSALU, IsSOP2];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    OneSGPROrSRegType:$dst0
  );
  let inputs = (ins
    OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
    OneSGPROrSRegOrIntConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src1
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], "${mnemonic} $dst0, $src0, $src1">
  ];
}
#endif // SOP2OUT1IN2INST_DEF

#ifndef SBFMB32_DEF
#define SBFMB32_DEF
def SBfmB32 : SOP2Out1In2Inst<"s_bfm_b32"> {
  let description = [{
    Calculate a bitfield mask given a field offset and size and store the
    result in a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0100010>];
}
#endif // SBFMB32_DEF

#ifndef SMULHII32_DEF
#define SMULHII32_DEF
def SMulHiI32 : SOP2Out1In2Inst<"s_mul_hi_i32"> {
  let description = [{
    Multiply two signed integers and store the high 32 bits of the result into
    a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0101101>];
}
#endif // SMULHII32_DEF

#ifndef SMULHIU32_DEF
#define SMULHIU32_DEF
def SMulHiU32 : SOP2Out1In2Inst<"s_mul_hi_u32"> {
  let description = [{
    Multiply two unsigned integers and store the high 32 bits of the result
    into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0101100>];
}
#endif // SMULHIU32_DEF

#ifndef SMULI32_DEF
#define SMULI32_DEF
def SMulI32 : SOP2Out1In2Inst<"s_mul_i32"> {
  let description = [{
    Multiply two signed 32-bit integer inputs and store the result into a
    scalar register.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0100100>];
}
#endif // SMULI32_DEF

#ifndef SOP2OUT1IN3INST_DEF
#define SOP2OUT1IN3INST_DEF
class SOP2Out1In3Inst<string mnemonic, list<Type> dst0Type, list<Type> src0Type, list<Type> src1Type> : AMDISAInstruction<mnemonic> {
  let props = [IsSALU, IsSOP2];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dst0
  );
  let inputs = (ins
    AnyOperandOf<src0Type>:$src0,
    AnyOperandOf<src1Type>:$src1,
    SCCType:$scc_src
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], "${mnemonic} $dst0, $src0, $src1">
  ];
}
#endif // SOP2OUT1IN3INST_DEF

#ifndef SCSELECTB32_DEF
#define SCSELECTB32_DEF
def SCselectB32 : SOP2Out1In3Inst<"s_cselect_b32",
    /*dst0=*/[OneSGPROrSRegType],
    /*src0=*/[OneSGPROrSRegOrLitConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>, F32, IntType<32>>],
    /*src1=*/[OneSGPROrSRegOrLitConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>, F32, IntType<32>>]
  > {
  let description = [{
    Select the first input if SCC is true otherwise select the second input,
    then store the selected input into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0001010>];
}
#endif // SCSELECTB32_DEF

#ifndef SCSELECTB64_DEF
#define SCSELECTB64_DEF
def SCselectB64 : SOP2Out1In3Inst<"s_cselect_b64",
    /*dst0=*/[TwoSGPROrSRegType],
    /*src0=*/[TwoSGPROrSRegOrConstType],
    /*src1=*/[TwoSGPROrSRegOrConstType]
  > {
  let description = [{
    Select the first input if SCC is true otherwise select the second input,
    then store the selected input into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SOP2_CDNA3, ENC_SOP2_CDNA4], 0b0001011>];
}
#endif // SCSELECTB64_DEF

#ifndef SMOVB32_DEF
#define SMOVB32_DEF
def SMovB32 : AMDISAInstruction<"s_mov_b32"> {
  let props = [IsSALU, IsSOP1];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Move scalar input into a scalar register.
  }];
  let encodings = [InstEnc<[ENC_SOP1_CDNA3, ENC_SOP1_CDNA4], 0b00000000>];
  let outputs = (ins
    OneSGPROrSRegType:$dst0
  );
  let inputs = (ins
    OneSGPROrSRegOrLitConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>, F32, IntType<32>>:$src0
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_SOP1_CDNA3, ENC_SOP1_CDNA4], "${mnemonic} $dst0, $src0">
  ];
}
#endif // SMOVB32_DEF

#ifndef SNOP_DEF
#define SNOP_DEF
def SNop : AMDISAInstruction<"s_nop"> {
  let props = [IsSALU, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Do nothing. Delay issue of next instruction by a small, fixed amount.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0000000>];
  let outputs = (ins);
  let inputs = (ins);
  let trailingArgs = (ins
    Imm16Attr:$imm
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    $imm attr-dict
  }];
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic} $imm">
  ];
}
#endif // SNOP_DEF

#ifndef SSETPRIO_DEF
#define SSETPRIO_DEF
def SSetprio : AMDISAInstruction<"s_setprio"> {
  let props = [IsSALU, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Change wave user priority.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0001111>];
  let outputs = (ins);
  let inputs = (ins);
  let trailingArgs = (ins
    Imm16Attr:$imm
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    $imm attr-dict
  }];
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic} $imm">
  ];
}
#endif // SSETPRIO_DEF


#ifndef SSLEEP_DEF
#define SSLEEP_DEF
def SSleep : AMDISAInstruction<"s_sleep"> {
  let props = [IsSALU, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Cause a wave to sleep for up to ~8000 clocks.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0001110>];
  let outputs = (ins);
  let inputs = (ins);
  let trailingArgs = (ins
    Imm16Attr:$imm
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    $imm attr-dict
  }];
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic} $imm">
  ];
}
#endif // SSLEEP_DEF

#ifndef STRAP_DEF
#define STRAP_DEF
def STrap : AMDISAInstruction<"s_trap"> {
  let props = [IsSALU, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Enter the trap handler.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0010010>];
  let outputs = (ins);
  let inputs = (ins);
  let trailingArgs = (ins
    Imm16Attr:$imm
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    $imm attr-dict
  }];
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic} $imm">
  ];
}
#endif // STRAP_DEF

#ifndef SWAITCNT_DEF
#define SWAITCNT_DEF
def SWaitcnt : AMDISAInstruction<"s_waitcnt"> {
  let props = [IsSALU, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Wait for the counts of outstanding local data share, vector memory and
    export instructions to be at or below the specified levels.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0001100>];
  let outputs = (ins);
  let trailingArgs = (ins
    VmcntAttr:$vmcnt,
    ExpcntAttr:$expcnt,
    LgkmcntAttr:$lgkmcnt,
    UnitAttr:$immutable
  );
  let genInstAssemblyFormat = 0;
  let assemblyFormat = [{
    (`vmcnt` `=` $vmcnt^)?
    (`expcnt` `=` $expcnt^)?
    (`lgkmcnt` `=` $lgkmcnt^)?
    (`immutable` $immutable^)?
    attr-dict
  }];
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic} $vmcnt $expcnt $lgkmcnt">
  ];
}
#endif // SWAITCNT_DEF

#ifndef SWAKEUP_DEF
#define SWAKEUP_DEF
def SWakeup : AMDISAInstruction<"s_wakeup"> {
  let props = [IsSALU, IsSOPP];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Allow a wave to 'ping' all the other waves in its threadgroup to force
    them to wake up early from an S_SLEEP instruction.
  }];
  let encodings = [InstEnc<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], 0b0000011>];
  let outputs = (ins);
  let inputs = (ins);
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_SOPP_CDNA3, ENC_SOPP_CDNA4], "${mnemonic}">
  ];
}
#endif // SWAKEUP_DEF

#endif // AMDGCN_VOP_TD
