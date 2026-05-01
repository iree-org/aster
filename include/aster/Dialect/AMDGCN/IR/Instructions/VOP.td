//===- VOP.td - AMDGCN VOP Instructions --------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_VOP_TD
#define AMDGCN_VOP_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

#ifndef VOP3OUT1IN3INST_DEF
#define VOP3OUT1IN3INST_DEF
class VOP3Out1In3Inst<string mnemonic, list<Type> dst0Type, list<Type> src0Type, list<Type> src2Type> : AMDISAInstruction<mnemonic> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dst0
  );
  let inputs = (ins
    AnyOperandOf<src0Type>:$src0,
    OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1,
    AnyOperandOf<src2Type>:$src2
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VOP3OUT1IN3INST_DEF

#ifndef VADD3U32_DEF
#define VADD3U32_DEF
def VAdd3U32 : VOP3Out1In3Inst<"v_add3_u32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src2=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Add three unsigned inputs and store the result into a vector register. No
    carry-in or carry-out support.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0111111111>];
}
#endif // VADD3U32_DEF

#ifndef VLSHLADDU64_DEF
#define VLSHLADDU64_DEF
def VLshlAddU64 : VOP3Out1In3Inst<"v_lshl_add_u64",
    /*dst0=*/[VGPROf<2>],
    /*src0=*/[TwoGPROrSRegOrIntLitType],
    /*src2=*/[TwoGPROrSRegOrIntLitType]
  > {
  let description = [{
    Given a shift count in the second input, calculate the logical shift left
    of the first input, then add the third input to the intermediate result,
    then store the final result into a vector register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1000001000>];
}
#endif // VLSHLADDU64_DEF

#ifndef VOP2OUT2IN3INST_DEF
#define VOP2OUT2IN3INST_DEF
class VOP2Out2In3Inst<string mnemonic> : AMDISAInstruction<mnemonic> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    VGPROf<1>:$dst0,
    AnyRegOf<[VCCType, TwoSGPROrVCCType]>:$dst1
  );
  let inputs = (ins
    AnyOperandOf<[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>, OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]>:$src0,
    AnyOperandOf<[VGPROf<1>, OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]>:$src1,
    AnyRegOf<[VCCType, TwoSGPROrVCCType]>:$src2
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], "${mnemonic} $dst0, $dst1, $src0, $src1, $src2">,
    ASMString<[ENC_VOP3_CDNA3], "${mnemonic}_e64 $dst0, $dst1, $src0, $src1, $src2">
  ];
}
#endif // VOP2OUT2IN3INST_DEF

#ifndef VADDCCOU32_DEF
#define VADDCCOU32_DEF
def VAddcCoU32 : VOP2Out2In3Inst<"v_addc_co_u32"> {
  let description = [{
    Add two unsigned 32-bit integer inputs and a bit from a carry-in mask,
    store the result into a vector register and store the carry-out mask into
    a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b011100, (pred
      VCCType:$dst1,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1,
      VCCType:$src2
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100011100, (pred
      TwoSGPROrVCCType:$dst1,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1,
      TwoSGPROrVCCType:$src2
    )>
  ];
}
#endif // VADDCCOU32_DEF

#ifndef VSUBBREVCOU32_DEF
#define VSUBBREVCOU32_DEF
def VSubbrevCoU32 : VOP2Out2In3Inst<"v_subbrev_co_u32"> {
  let description = [{
    Subtract the first unsigned 32-bit integer input from the second input,
    subtract a bit from the carry-in mask, store the result into a vector
    register and store the carry-out mask into a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b011110, (pred
      VCCType:$dst1,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1,
      VCCType:$src2
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100011110, (pred
      TwoSGPROrVCCType:$dst1,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1,
      TwoSGPROrVCCType:$src2
    )>
  ];
}
#endif // VSUBBREVCOU32_DEF

#ifndef VSUBBCOU32_DEF
#define VSUBBCOU32_DEF
def VSubbCoU32 : VOP2Out2In3Inst<"v_subb_co_u32"> {
  let description = [{
    Subtract the second unsigned 32-bit integer input from the first input,
    subtract a bit from the carry-in mask, store the result into a vector
    register and store the carry-out mask into a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b011101, (pred
      VCCType:$dst1,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1,
      VCCType:$src2
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100011101, (pred
      TwoSGPROrVCCType:$dst1,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1,
      TwoSGPROrVCCType:$src2
    )>
  ];
}
#endif // VSUBBCOU32_DEF

#ifndef VOP2OUT2IN2INST_DEF
#define VOP2OUT2IN2INST_DEF
class VOP2Out2In2Inst<string mnemonic> : AMDISAInstruction<mnemonic> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    VGPROf<1>:$dst0,
    AnyRegOf<[VCCType, TwoSGPROrVCCType]>:$dst1
  );
  let inputs = (ins
    AnyOperandOf<[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>, OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]>:$src0,
    AnyOperandOf<[VGPROf<1>, OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]>:$src1
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], "${mnemonic} $dst0, $dst1, $src0, $src1">,
    ASMString<[ENC_VOP3_CDNA3], "${mnemonic}_e64 $dst0, $dst1, $src0, $src1">
  ];
}
#endif // VOP2OUT2IN2INST_DEF

#ifndef VADDCOU32_DEF
#define VADDCOU32_DEF
def VAddCoU32 : VOP2Out2In2Inst<"v_add_co_u32"> {
  let description = [{
    Add two unsigned 32-bit integer inputs, store the result into a vector
    register and store the carry-out mask into a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b011001, (pred
      VCCType:$dst1,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100011001, (pred
      TwoSGPROrVCCType:$dst1,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VADDCOU32_DEF

#ifndef VSUBREVCOU32_DEF
#define VSUBREVCOU32_DEF
def VSubrevCoU32 : VOP2Out2In2Inst<"v_subrev_co_u32"> {
  let description = [{
    Subtract the first unsigned 32-bit integer input from the second input,
    store the result into a vector register and store the carry-out mask into
    a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b011011, (pred
      VCCType:$dst1,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100011011, (pred
      TwoSGPROrVCCType:$dst1,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VSUBREVCOU32_DEF

#ifndef VSUBCOU32_DEF
#define VSUBCOU32_DEF
def VSubCoU32 : VOP2Out2In2Inst<"v_sub_co_u32"> {
  let description = [{
    Subtract the second unsigned 32-bit integer input from the first input,
    store the result into a vector register and store the carry-out mask into
    a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b011010, (pred
      VCCType:$dst1,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100011010, (pred
      TwoSGPROrVCCType:$dst1,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VSUBCOU32_DEF

#ifndef VOP2OUT1IN2INST_DEF
#define VOP2OUT1IN2INST_DEF
class VOP2Out1In2Inst<string mnemonic, list<Type> src0Type, list<Type> src1Type> : AMDISAInstruction<mnemonic> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    VGPROf<1>:$dst0
  );
  let inputs = (ins
    AnyOperandOf<src0Type>:$src0,
    AnyOperandOf<src1Type>:$src1
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], "${mnemonic} $dst0, $src0, $src1">,
    ASMString<[ENC_VOP3_CDNA3], "${mnemonic}_e64 $dst0, $src0, $src1">
  ];
}
#endif // VOP2OUT1IN2INST_DEF

#ifndef VADDF16_DEF
#define VADDF16_DEF
def VAddF16 : VOP2Out1In2Inst<"v_add_f16",
    /*src0=*/[OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>]
  > {
  let description = [{
    Add two floating point inputs and store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b011111, (pred
      OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100011111, (pred
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src0,
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src1
    )>
  ];
}
#endif // VADDF16_DEF

#ifndef VADDF32_DEF
#define VADDF32_DEF
def VAddF32 : VOP2Out1In2Inst<"v_add_f32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Add two floating point inputs and store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b000001, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100000001, (pred
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src1
    )>
  ];
}
#endif // VADDF32_DEF

#ifndef VADDU16_DEF
#define VADDU16_DEF
def VAddU16 : VOP2Out1In2Inst<"v_add_u16",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>]
  > {
  let description = [{
    Add two unsigned 16-bit integer inputs and store the result into a vector
    register. No carry-in or carry-out support.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b100110, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100100110, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src1
    )>
  ];
}
#endif // VADDU16_DEF

#ifndef VADDU32_DEF
#define VADDU32_DEF
def VAddU32 : VOP2Out1In2Inst<"v_add_u32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Add two unsigned 32-bit integer inputs and store the result into a vector
    register. No carry-in or carry-out support.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b110100, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100110100, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VADDU32_DEF

#ifndef VANDB32_DEF
#define VANDB32_DEF
def VAndB32 : VOP2Out1In2Inst<"v_and_b32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Calculate bitwise AND on two vector inputs and store the result into a
    vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b010011, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100010011, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VANDB32_DEF

#ifndef VASHRREVI16_DEF
#define VASHRREVI16_DEF
def VAshrrevI16 : VOP2Out1In2Inst<"v_ashrrev_i16",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the arithmetic
    shift right (preserving sign bit) of the second vector input and store the
    result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b101100, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100101100, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src1
    )>
  ];
}
#endif // VASHRREVI16_DEF

#ifndef VASHRREVI32_DEF
#define VASHRREVI32_DEF
def VAshrrevI32 : VOP2Out1In2Inst<"v_ashrrev_i32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the arithmetic
    shift right (preserving sign bit) of the second vector input and store the
    result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b010001, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100010001, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VASHRREVI32_DEF

#ifndef VLSHLREVB16_DEF
#define VLSHLREVB16_DEF
def VLshlrevB16 : VOP2Out1In2Inst<"v_lshlrev_b16",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the logical shift
    left of the second vector input and store the result into a vector
    register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b101010, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100101010, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src1
    )>
  ];
}
#endif // VLSHLREVB16_DEF

#ifndef VLSHLREVB32_DEF
#define VLSHLREVB32_DEF
def VLshlrevB32 : VOP2Out1In2Inst<"v_lshlrev_b32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the logical shift
    left of the second vector input and store the result into a vector
    register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b010010, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100010010, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VLSHLREVB32_DEF

#ifndef VLSHRREVB16_DEF
#define VLSHRREVB16_DEF
def VLshrrevB16 : VOP2Out1In2Inst<"v_lshrrev_b16",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the logical shift
    right of the second vector input and store the result into a vector
    register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b101011, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100101011, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src1
    )>
  ];
}
#endif // VLSHRREVB16_DEF

#ifndef VLSHRREVB32_DEF
#define VLSHRREVB32_DEF
def VLshrrevB32 : VOP2Out1In2Inst<"v_lshrrev_b32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the logical shift
    right of the second vector input and store the result into a vector
    register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b010000, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100010000, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VLSHRREVB32_DEF

#ifndef VMAXF32_DEF
#define VMAXF32_DEF
def VMaxF32 : VOP2Out1In2Inst<"v_max_f32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Select the maximum of two single-precision float inputs and store the
    selected value into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b001011, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100001011, (pred
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src1
    )>
  ];
}
#endif // VMAXF32_DEF

#ifndef VMINF32_DEF
#define VMINF32_DEF
def VMinF32 : VOP2Out1In2Inst<"v_min_f32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Select the minimum of two single-precision float inputs and store the
    selected value into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b001010, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100001010, (pred
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src1
    )>
  ];
}
#endif // VMINF32_DEF

#ifndef VMULF16_DEF
#define VMULF16_DEF
def VMulF16 : VOP2Out1In2Inst<"v_mul_f16",
    /*src0=*/[OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>]
  > {
  let description = [{
    Multiply two floating point inputs and store the result into a vector
    register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b100010, (pred
      OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100100010, (pred
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src0,
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src1
    )>
  ];
}
#endif // VMULF16_DEF

#ifndef VMULF32_DEF
#define VMULF32_DEF
def VMulF32 : VOP2Out1In2Inst<"v_mul_f32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply two floating point inputs and store the result into a vector
    register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b000101, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100000101, (pred
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src1
    )>
  ];
}
#endif // VMULF32_DEF

#ifndef VMULLOU16_DEF
#define VMULLOU16_DEF
def VMulLoU16 : VOP2Out1In2Inst<"v_mul_lo_u16",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>]
  > {
  let description = [{
    Multiply two unsigned 16-bit integer inputs and store the low bits of the
    result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b101001, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100101001, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src1
    )>
  ];
}
#endif // VMULLOU16_DEF

#ifndef VNOP_DEF
#define VNOP_DEF
def VNop : AMDISAInstruction<"v_nop"> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Do nothing.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00000000>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0110000000>
  ];
  let outputs = (ins);
  let inputs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], "${mnemonic}">,
    ASMString<[ENC_VOP3_CDNA3], "${mnemonic}_e64">
  ];
}
#endif // VNOP_DEF

#ifndef VORB32_DEF
#define VORB32_DEF
def VOrB32 : VOP2Out1In2Inst<"v_or_b32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Calculate bitwise OR on two vector inputs and store the result into a
    vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b010100, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100010100, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VORB32_DEF

#ifndef VSUBREVF16_DEF
#define VSUBREVF16_DEF
def VSubrevF16 : VOP2Out1In2Inst<"v_subrev_f16",
    /*src0=*/[OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>]
  > {
  let description = [{
    Subtract the first floating point input from the second input and store
    the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b100001, (pred
      OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100100001, (pred
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src0,
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src1
    )>
  ];
}
#endif // VSUBREVF16_DEF

#ifndef VSUBREVU32_DEF
#define VSUBREVU32_DEF
def VSubrevU32 : VOP2Out1In2Inst<"v_subrev_u32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Subtract the first unsigned 32-bit integer input from the second input and
    store the result into a vector register. No carry-in or carry-out support.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b110110, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100110110, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VSUBREVU32_DEF

#ifndef VSUBF16_DEF
#define VSUBF16_DEF
def VSubF16 : VOP2Out1In2Inst<"v_sub_f16",
    /*src0=*/[OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>]
  > {
  let description = [{
    Subtract the second floating point input from the first input and store
    the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b100000, (pred
      OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100100000, (pred
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src0,
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src1
    )>
  ];
}
#endif // VSUBF16_DEF

#ifndef VSUBF32_DEF
#define VSUBF32_DEF
def VSubF32 : VOP2Out1In2Inst<"v_sub_f32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Subtract the second floating point input from the first input and store
    the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b000010, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100000010, (pred
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src1
    )>
  ];
}
#endif // VSUBF32_DEF

#ifndef VSUBU16_DEF
#define VSUBU16_DEF
def VSubU16 : VOP2Out1In2Inst<"v_sub_u16",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>]
  > {
  let description = [{
    Subtract the second unsigned 16-bit integer input from the first input and
    store the result into a vector register. No carry-in or carry-out support.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b100111, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<16>]>, IntType<16>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100100111, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>:$src1
    )>
  ];
}
#endif // VSUBU16_DEF

#ifndef VSUBU32_DEF
#define VSUBU32_DEF
def VSubU32 : VOP2Out1In2Inst<"v_sub_u32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Subtract the second unsigned 32-bit integer input from the first input and
    store the result into a vector register. No carry-in or carry-out support.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b110101, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100110101, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VSUBU32_DEF

#ifndef VXORB32_DEF
#define VXORB32_DEF
def VXorB32 : VOP2Out1In2Inst<"v_xor_b32",
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[VGPROf<1>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Calculate bitwise XOR on two vector inputs and store the result into a
    vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b010101, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100010101, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VXORB32_DEF

#ifndef VOP3OUT1IN2INST_DEF
#define VOP3OUT1IN2INST_DEF
class VOP3Out1In2Inst<string mnemonic, list<Type> dst0Type, list<Type> src0Type, list<Type> src1Type> : AMDISAInstruction<mnemonic> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dst0
  );
  let inputs = (ins
    AnyOperandOf<src0Type>:$src0,
    AnyOperandOf<src1Type>:$src1
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], "${mnemonic} $dst0, $src0, $src1">
  ];
}
#endif // VOP3OUT1IN2INST_DEF

#ifndef VADDI16_DEF
#define VADDI16_DEF
def VAddI16 : VOP3Out1In2Inst<"v_add_i16",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>],
    /*src1=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<16>]>>]
  > {
  let description = [{
    Add two signed 16-bit integer inputs and store the result into a vector
    register. No carry-in or carry-out support.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010011110>];
}
#endif // VADDI16_DEF

#ifndef VADDI32_DEF
#define VADDI32_DEF
def VAddI32 : VOP3Out1In2Inst<"v_add_i32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Add two signed 32-bit integer inputs and store the result into a vector
    register. No carry-in or carry-out support.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010011100>];
}
#endif // VADDI32_DEF

#ifndef VASHRREVI64_DEF
#define VASHRREVI64_DEF
def VAshrrevI64 : VOP3Out1In2Inst<"v_ashrrev_i64",
    /*dst0=*/[VGPROf<2>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[TwoGPROrSRegOrIntLitType]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the arithmetic
    shift right (preserving sign bit) of the second vector input and store the
    result into a vector register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010010001>];
}
#endif // VASHRREVI64_DEF

#ifndef VCVTPKBF8F32_DEF
#define VCVTPKBF8F32_DEF
def VCvtPkBf8F32 : VOP3Out1In2Inst<"v_cvt_pk_bf8_f32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>],
    /*src1=*/[OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Convert from two single-precision float inputs to a packed BF8 float value
    with round to nearest even semantics and store the result into 16 bits of
    a vector register using OPSEL.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010100011>];
}
#endif // VCVTPKBF8F32_DEF

#ifndef VCVTPKFP8F32_DEF
#define VCVTPKFP8F32_DEF
def VCvtPkFp8F32 : VOP3Out1In2Inst<"v_cvt_pk_fp8_f32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>],
    /*src1=*/[OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Convert from two single-precision float inputs to a packed FP8 float value
    with round to nearest even semantics and store the result into 16 bits of
    a vector register using OPSEL.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010100010>];
}
#endif // VCVTPKFP8F32_DEF

#ifndef VLSHLREVB64_DEF
#define VLSHLREVB64_DEF
def VLshlrevB64 : VOP3Out1In2Inst<"v_lshlrev_b64",
    /*dst0=*/[VGPROf<2>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[TwoGPROrSRegOrIntLitType]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the logical shift
    left of the second vector input and store the result into a vector
    register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010001111>];
}
#endif // VLSHLREVB64_DEF

#ifndef VLSHRREVB64_DEF
#define VLSHRREVB64_DEF
def VLshrrevB64 : VOP3Out1In2Inst<"v_lshrrev_b64",
    /*dst0=*/[VGPROf<2>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[TwoGPROrSRegOrIntLitType]
  > {
  let description = [{
    Given a shift count in the first vector input, calculate the logical shift
    right of the second vector input and store the result into a vector
    register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010010000>];
}
#endif // VLSHRREVB64_DEF

#ifndef VMULHII32_DEF
#define VMULHII32_DEF
def VMulHiI32 : VOP3Out1In2Inst<"v_mul_hi_i32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Multiply two signed 32-bit integer inputs and store the high 32 bits of
    the result into a vector register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010000111>];
}
#endif // VMULHII32_DEF

#ifndef VMULHIU32_DEF
#define VMULHIU32_DEF
def VMulHiU32 : VOP3Out1In2Inst<"v_mul_hi_u32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Multiply two unsigned 32-bit integer inputs and store the high 32 bits of
    the result into a vector register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010000110>];
}
#endif // VMULHIU32_DEF

#ifndef VMULLOU32_DEF
#define VMULLOU32_DEF
def VMulLoU32 : VOP3Out1In2Inst<"v_mul_lo_u32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>],
    /*src1=*/[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Multiply two unsigned 32-bit integer inputs and store the result into a
    vector register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010000101>];
}
#endif // VMULLOU32_DEF

#ifndef VPACKB32F16_DEF
#define VPACKB32F16_DEF
def VPackB32F16 : VOP3Out1In2Inst<"v_pack_b32_f16",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>],
    /*src1=*/[OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>]
  > {
  let description = [{
    Pack two half-precision float values into a single 32-bit value and store
    the result into a vector register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b1010100000>];
}
#endif // VPACKB32F16_DEF

#ifndef VCMPINST_DEF
#define VCMPINST_DEF
class VCmpInst<string mnemonic> : AMDISAInstruction<mnemonic> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    AnyRegOf<[TwoSGPROrVCCType, VCCType]>:$dst0
  );
  let inputs = (ins
    AnyOperandOf<[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>, OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>]>:$src0,
    AnyOperandOf<[OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>, VGPROf<1>]>:$src1
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP3_CDNA3], "${mnemonic}_e64 $dst0, $src0, $src1">,
    ASMString<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4, ENC_VOP3_CDNA4], "${mnemonic} $dst0, $src0, $src1">
  ];
}
#endif // VCMPINST_DEF

#ifndef VCMPEQI32_DEF
#define VCMPEQI32_DEF
def VCmpEqI32 : VCmpInst<"v_cmp_eq_i32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is equal to the
    second input. Store the result into VCC or a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11000010, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011000010, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPEQI32_DEF

#ifndef VCMPGEI32_DEF
#define VCMPGEI32_DEF
def VCmpGeI32 : VCmpInst<"v_cmp_ge_i32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is greater than
    or equal to the second input. Store the result into VCC or a scalar
    register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11000110, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011000110, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPGEI32_DEF

#ifndef VCMPGEU32_DEF
#define VCMPGEU32_DEF
def VCmpGeU32 : VCmpInst<"v_cmp_ge_u32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is greater than
    or equal to the second input. Store the result into VCC or a scalar
    register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11001110, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011001110, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPGEU32_DEF

#ifndef VCMPGTI32_DEF
#define VCMPGTI32_DEF
def VCmpGtI32 : VCmpInst<"v_cmp_gt_i32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is greater than
    the second input. Store the result into VCC or a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11000100, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011000100, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPGTI32_DEF

#ifndef VCMPGTU32_DEF
#define VCMPGTU32_DEF
def VCmpGtU32 : VCmpInst<"v_cmp_gt_u32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is greater than
    the second input. Store the result into VCC or a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11001100, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011001100, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPGTU32_DEF

#ifndef VCMPLEI32_DEF
#define VCMPLEI32_DEF
def VCmpLeI32 : VCmpInst<"v_cmp_le_i32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is less than or
    equal to the second input. Store the result into VCC or a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11000011, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011000011, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPLEI32_DEF

#ifndef VCMPLEU32_DEF
#define VCMPLEU32_DEF
def VCmpLeU32 : VCmpInst<"v_cmp_le_u32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is less than or
    equal to the second input. Store the result into VCC or a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11001011, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011001011, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPLEU32_DEF

#ifndef VCMPLTI32_DEF
#define VCMPLTI32_DEF
def VCmpLtI32 : VCmpInst<"v_cmp_lt_i32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is less than the
    second input. Store the result into VCC or a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11000001, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011000001, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPLTI32_DEF

#ifndef VCMPLTU32_DEF
#define VCMPLTU32_DEF
def VCmpLtU32 : VCmpInst<"v_cmp_lt_u32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is less than the
    second input. Store the result into VCC or a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11001001, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011001001, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPLTU32_DEF

#ifndef VCMPNEI32_DEF
#define VCMPNEI32_DEF
def VCmpNeI32 : VCmpInst<"v_cmp_ne_i32"> {
  let description = [{
    Set the per-lane condition code to 1 iff the first input is not equal to
    the second input. Store the result into VCC or a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOPC_CDNA3, ENC_VOPC_CDNA4], 0b11000101, (pred
      VCCType:$dst0,
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0,
      VGPROf<1>:$src1
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0011000101, (pred
      TwoSGPROrVCCType:$dst0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1
    )>
  ];
}
#endif // VCMPNEI32_DEF

#ifndef VCNDMASKB32_DEF
#define VCNDMASKB32_DEF
def VCndmaskB32 : AMDISAInstruction<"v_cndmask_b32"> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Copy data from one of two inputs based on the per-lane condition code and
    store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], 0b000000, (pred
      OneGPROrSRegOrConstType:$src0,
      VGPROf<1>:$src1,
      VCCType:$src2
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0100000000, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>:$src0,
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>:$src1,
      TwoSGPROrVCCType:$src2
    )>
  ];
  let outputs = (ins
    VGPROf<1>:$dst0
  );
  let inputs = (ins
    AnyOperandOf<[OneGPROrSRegOrConstType, OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>]>:$src0,
    AnyOperandOf<[VGPROf<1>, OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>]>:$src1,
    AnyRegOf<[VCCType, TwoSGPROrVCCType]>:$src2
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP2_CDNA3, ENC_VOP2_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">,
    ASMString<[ENC_VOP3_CDNA3], "${mnemonic}_e64 $dst0, $src0, $src1, $src2">
  ];
}
#endif // VCNDMASKB32_DEF

#ifndef VOP1OUT1IN1INST_DEF
#define VOP1OUT1IN1INST_DEF
class VOP1Out1In1Inst<string mnemonic, list<Type> dst0Type, list<Type> src0Type> : AMDISAInstruction<mnemonic> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dst0
  );
  let inputs = (ins
    AnyOperandOf<src0Type>:$src0
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], "${mnemonic} $dst0, $src0">,
    ASMString<[ENC_VOP3_CDNA3], "${mnemonic}_e64 $dst0, $src0">
  ];
}
#endif // VOP1OUT1IN1INST_DEF

#ifndef VCVTF16F32_DEF
#define VCVTF16F32_DEF
def VCvtF16F32 : VOP1Out1In1Inst<"v_cvt_f16_f32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Convert from a single-precision float input to a half-precision float
    value and store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00001010, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>:$src0
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0101001010, (pred
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src0
    )>
  ];
}
#endif // VCVTF16F32_DEF

#ifndef VCVTF32F16_DEF
#define VCVTF32F16_DEF
def VCvtF32F16 : VOP1Out1In1Inst<"v_cvt_f32_f16",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>,
             OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>]
  > {
  let description = [{
    Convert from a half-precision float input to a single-precision float
    value and store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00001011, (pred
      OneGPROrVCCM0OrFPConstOf<InlineLitFP<[F16]>, F16>:$src0
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0101001011, (pred
      OneGPROrVCCM0OrFPLitOf<InlineLitFP<[F16]>>:$src0
    )>
  ];
}
#endif // VCVTF32F16_DEF

#ifndef VCVTF32I32_DEF
#define VCVTF32I32_DEF
def VCvtF32I32 : VOP1Out1In1Inst<"v_cvt_f32_i32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Convert from a signed 32-bit integer input to a single-precision float
    value and store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00000101, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0101000101, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0
    )>
  ];
}
#endif // VCVTF32I32_DEF

#ifndef VCVTF32U32_DEF
#define VCVTF32U32_DEF
def VCvtF32U32 : VOP1Out1In1Inst<"v_cvt_f32_u32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>,
             OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Convert from an unsigned 32-bit integer input to a single-precision float
    value and store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00000110, (pred
      OneGPROrSRegOrConstOf<InlineLitInt<[IntType<32>]>, IntType<32>>:$src0
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0101000110, (pred
      OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0
    )>
  ];
}
#endif // VCVTF32U32_DEF

#ifndef VCVTI32F32_DEF
#define VCVTI32F32_DEF
def VCvtI32F32 : VOP1Out1In1Inst<"v_cvt_i32_f32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Convert from a single-precision float input to a signed 32-bit integer
    value and store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00001000, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>:$src0
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0101001000, (pred
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src0
    )>
  ];
}
#endif // VCVTI32F32_DEF

#ifndef VCVTU32F32_DEF
#define VCVTU32F32_DEF
def VCvtU32F32 : VOP1Out1In1Inst<"v_cvt_u32_f32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>,
             OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>]
  > {
  let description = [{
    Convert from a single-precision float input to an unsigned 32-bit integer
    value and store the result into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00000111, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, F32>:$src0
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0101000111, (pred
      OneGPROrSRegOrLitOf<InlineLitFP<[F32]>>:$src0
    )>
  ];
}
#endif // VCVTU32F32_DEF

#ifndef VMOVB32_DEF
#define VMOVB32_DEF
def VMovB32 : VOP1Out1In1Inst<"v_mov_b32",
    /*dst0=*/[VGPROf<1>],
    /*src0=*/[OneGPROrSRegOrConstType,
             OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>]
  > {
  let description = [{
    Move 32-bit data from a vector input into a vector register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00000001, (pred
      OneGPROrSRegOrConstType:$src0
    )>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0101000001, (pred
      OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>:$src0
    )>
  ];
}
#endif // VMOVB32_DEF

#ifndef VREADFIRSTLANEB32_DEF
#define VREADFIRSTLANEB32_DEF
def VReadfirstlaneB32 : VOP1Out1In1Inst<"v_readfirstlane_b32",
    /*dst0=*/[SGPROf<1>],
    /*src0=*/[VGPROf<1>]
  > {
  let description = [{
    Read the scalar value in the lowest active lane of the input vector
    register and store it into a scalar register.
  }];
  let encodings = [
    InstEnc<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], 0b00000010>,
    InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0101000010>
  ];
  let asm_syntax = [
    ASMString<[ENC_VOP1_CDNA3, ENC_VOP1_CDNA4], "${mnemonic} $dst0, $src0">
  ];
}
#endif // VREADFIRSTLANEB32_DEF

#ifndef VMADU64U32_DEF
#define VMADU64U32_DEF
def VMadU64U32 : AMDISAInstruction<"v_mad_u64_u32"> {
  let props = [IsVALU];
  let archs = [CDNA3Isa, CDNA4Isa];
  let description = [{
    Multiply two unsigned integer inputs, add a third unsigned integer input,
    store the result into a 64-bit vector register and store the
    overflow/carryout into a scalar mask register.
  }];
  let encodings = [InstEnc<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], 0b0111101000>];
  let outputs = (ins
    VGPROf<2>:$dst0,
    TwoSGPROrVCCType:$dst1
  );
  let inputs = (ins
    OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src0,
    OneGPROrSRegOrLitOf<InlineLitInt<[IntType<32>]>>:$src1,
    TwoGPROrSRegOrIntLitType:$src2
  );
  let trailingArgs = (ins);
  let asm_syntax = [
    ASMString<[ENC_VOP3_CDNA3, ENC_VOP3_CDNA4], "${mnemonic} $dst0, $dst1, $src0, $src1, $src2">
  ];
}
#endif // VMADU64U32_DEF

#endif // AMDGCN_VOP_TD
