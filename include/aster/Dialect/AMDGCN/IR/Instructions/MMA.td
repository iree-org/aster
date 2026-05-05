//===- MMA.td - AMDGCN MMA Instructions --------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_MMA_TD
#define AMDGCN_MMA_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/InstOpInterfaces.td"


#ifndef VMFMAINST_DEF
#define VMFMAINST_DEF
class VMfmaInst<string mnemonic, list<Type> dst0Type, list<Type> src0Type, list<Type> src1Type, list<Type> src2Type> : AMDISAInstruction<mnemonic, [MMAOpInterface]> {
  let props = [IsVALU, IsVOP3P, IsMMA];
  let archs = [CDNA3Isa, CDNA4Isa];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dst0
  );
  let inputs = (ins
    AnyRegOf<src0Type>:$src0,
    AnyRegOf<src1Type>:$src1,
    AnyOperandOf<src2Type>:$src2
  );
  let trailingArgs = (ins
    CBSZAttr:$cbsz,
    ABIDAttr:$abid,
    BLGPAttr:$blgp
  );
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2 $cbsz$abid$blgp">
  ];
}
#endif // VMFMAINST_DEF

#ifndef VMFMAF3216X16X128F8F6F4_DEF
#define VMFMAF3216X16X128F8F6F4_DEF
def VMfmaF3216x16x128F8f6f4 : VMfmaInst<"v_mfma_f32_16x16x128_f8f6f4",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<8>],
    /*src1=*/[VGPROrAGPROf<8>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Multiply the 16x128 matrix in the first input by the 128x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA4], 0b0101101>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAF3216X16X128F8F6F4_DEF

#ifndef VMFMAF3216X16X16BF16_DEF
#define VMFMAF3216X16X16BF16_DEF
def VMfmaF3216x16x16Bf16 : VMfmaInst<"v_mfma_f32_16x16x16_bf16",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x16 matrix in the first input by the 16x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1100001>];
}
#endif // VMFMAF3216X16X16BF16_DEF

#ifndef VMFMAF3216X16X16F16_DEF
#define VMFMAF3216X16X16F16_DEF
def VMfmaF3216x16x16F16 : VMfmaInst<"v_mfma_f32_16x16x16_f16",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x16 matrix in the first input by the 16x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1001101>];
}
#endif // VMFMAF3216X16X16F16_DEF

#ifndef VMFMAF3216X16X14BF32_DEF
#define VMFMAF3216X16X14BF32_DEF
def VMfmaF3216x16x14bF32 : VMfmaInst<"v_mfma_f32_16x16x1_4b_f32",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<1>],
    /*src1=*/[VGPROrAGPROf<1>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x1 matrix in the first input by the 1x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1000001>];
}
#endif // VMFMAF3216X16X14BF32_DEF

#ifndef VMFMAF3216X16X32BF16_DEF
#define VMFMAF3216X16X32BF16_DEF
def VMfmaF3216x16x32Bf16 : VMfmaInst<"v_mfma_f32_16x16x32_bf16",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<4>],
    /*src1=*/[VGPROrAGPROf<4>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Multiply the 16x32 matrix in the first input by the 32x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA4], 0b0110101>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAF3216X16X32BF16_DEF

#ifndef VMFMAF3216X16X32BF8BF8_DEF
#define VMFMAF3216X16X32BF8BF8_DEF
def VMfmaF3216x16x32Bf8Bf8 : VMfmaInst<"v_mfma_f32_16x16x32_bf8_bf8",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x32 matrix in the first input by the 32x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1110000>];
}
#endif // VMFMAF3216X16X32BF8BF8_DEF

#ifndef VMFMAF3216X16X32BF8FP8_DEF
#define VMFMAF3216X16X32BF8FP8_DEF
def VMfmaF3216x16x32Bf8Fp8 : VMfmaInst<"v_mfma_f32_16x16x32_bf8_fp8",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x32 matrix in the first input by the 32x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1110001>];
}
#endif // VMFMAF3216X16X32BF8FP8_DEF

#ifndef VMFMAF3216X16X32F16_DEF
#define VMFMAF3216X16X32F16_DEF
def VMfmaF3216x16x32F16 : VMfmaInst<"v_mfma_f32_16x16x32_f16",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<4>],
    /*src1=*/[VGPROrAGPROf<4>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Multiply the 16x32 matrix in the first input by the 32x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA4], 0b1010100>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAF3216X16X32F16_DEF

#ifndef VMFMAF3216X16X32FP8BF8_DEF
#define VMFMAF3216X16X32FP8BF8_DEF
def VMfmaF3216x16x32Fp8Bf8 : VMfmaInst<"v_mfma_f32_16x16x32_fp8_bf8",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x32 matrix in the first input by the 32x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1110010>];
}
#endif // VMFMAF3216X16X32FP8BF8_DEF

#ifndef VMFMAF3216X16X32FP8FP8_DEF
#define VMFMAF3216X16X32FP8FP8_DEF
def VMfmaF3216x16x32Fp8Fp8 : VMfmaInst<"v_mfma_f32_16x16x32_fp8_fp8",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x32 matrix in the first input by the 32x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1110011>];
}
#endif // VMFMAF3216X16X32FP8FP8_DEF

#ifndef VMFMAF3216X16X44BBF16_DEF
#define VMFMAF3216X16X44BBF16_DEF
def VMfmaF3216x16x44bBf16 : VMfmaInst<"v_mfma_f32_16x16x4_4b_bf16",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x4 matrix in the first input by the 4x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1011110>];
}
#endif // VMFMAF3216X16X44BBF16_DEF

#ifndef VMFMAF3216X16X44BF16_DEF
#define VMFMAF3216X16X44BF16_DEF
def VMfmaF3216x16x44bF16 : VMfmaInst<"v_mfma_f32_16x16x4_4b_f16",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x4 matrix in the first input by the 4x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1001001>];
}
#endif // VMFMAF3216X16X44BF16_DEF

#ifndef VMFMAF3216X16X4F32_DEF
#define VMFMAF3216X16X4F32_DEF
def VMfmaF3216x16x4F32 : VMfmaInst<"v_mfma_f32_16x16x4_f32",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<1>],
    /*src1=*/[VGPROrAGPROf<1>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x4 matrix in the first input by the 4x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1000101>];
}
#endif // VMFMAF3216X16X4F32_DEF

#ifndef VMFMAF3216X16X8XF32_DEF
#define VMFMAF3216X16X8XF32_DEF
def VMfmaF3216x16x8Xf32 : VMfmaInst<"v_mfma_f32_16x16x8_xf32",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let archs = [CDNA3Isa];
  let description = [{
    Multiply the 16x8 matrix in the first input by the 8x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3], 0b0111110>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA3], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAF3216X16X8XF32_DEF

#ifndef VMFMAF3232X32X16BF16_DEF
#define VMFMAF3232X32X16BF16_DEF
def VMfmaF3232x32x16Bf16 : VMfmaInst<"v_mfma_f32_32x32x16_bf16",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<4>],
    /*src1=*/[VGPROrAGPROf<4>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Multiply the 32x16 matrix in the first input by the 16x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA4], 0b0110111>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAF3232X32X16BF16_DEF

#ifndef VMFMAF3232X32X16BF8BF8_DEF
#define VMFMAF3232X32X16BF8BF8_DEF
def VMfmaF3232x32x16Bf8Bf8 : VMfmaInst<"v_mfma_f32_32x32x16_bf8_bf8",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x16 matrix in the first input by the 16x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1110100>];
}
#endif // VMFMAF3232X32X16BF8BF8_DEF

#ifndef VMFMAF3232X32X16BF8FP8_DEF
#define VMFMAF3232X32X16BF8FP8_DEF
def VMfmaF3232x32x16Bf8Fp8 : VMfmaInst<"v_mfma_f32_32x32x16_bf8_fp8",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x16 matrix in the first input by the 16x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1110101>];
}
#endif // VMFMAF3232X32X16BF8FP8_DEF

#ifndef VMFMAF3232X32X16F16_DEF
#define VMFMAF3232X32X16F16_DEF
def VMfmaF3232x32x16F16 : VMfmaInst<"v_mfma_f32_32x32x16_f16",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<4>],
    /*src1=*/[VGPROrAGPROf<4>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Multiply the 32x16 matrix in the first input by the 16x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA4], 0b1010101>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAF3232X32X16F16_DEF

#ifndef VMFMAF3232X32X16FP8BF8_DEF
#define VMFMAF3232X32X16FP8BF8_DEF
def VMfmaF3232x32x16Fp8Bf8 : VMfmaInst<"v_mfma_f32_32x32x16_fp8_bf8",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x16 matrix in the first input by the 16x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1110110>];
}
#endif // VMFMAF3232X32X16FP8BF8_DEF

#ifndef VMFMAF3232X32X16FP8FP8_DEF
#define VMFMAF3232X32X16FP8FP8_DEF
def VMfmaF3232x32x16Fp8Fp8 : VMfmaInst<"v_mfma_f32_32x32x16_fp8_fp8",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x16 matrix in the first input by the 16x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1110111>];
}
#endif // VMFMAF3232X32X16FP8FP8_DEF

#ifndef VMFMAF3232X32X12BF32_DEF
#define VMFMAF3232X32X12BF32_DEF
def VMfmaF3232x32x12bF32 : VMfmaInst<"v_mfma_f32_32x32x1_2b_f32",
    /*dst0=*/[VGPROrAGPROf<32>],
    /*src0=*/[VGPROrAGPROf<1>],
    /*src1=*/[VGPROrAGPROf<1>],
    /*src2=*/[VGPROrAGPROrLitOf<32, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x1 matrix in the first input by the 1x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1000000>];
}
#endif // VMFMAF3232X32X12BF32_DEF

#ifndef VMFMAF3232X32X2F32_DEF
#define VMFMAF3232X32X2F32_DEF
def VMfmaF3232x32x2F32 : VMfmaInst<"v_mfma_f32_32x32x2_f32",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<1>],
    /*src1=*/[VGPROrAGPROf<1>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x2 matrix in the first input by the 2x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1000100>];
}
#endif // VMFMAF3232X32X2F32_DEF

#ifndef VMFMAF3232X32X42BBF16_DEF
#define VMFMAF3232X32X42BBF16_DEF
def VMfmaF3232x32x42bBf16 : VMfmaInst<"v_mfma_f32_32x32x4_2b_bf16",
    /*dst0=*/[VGPROrAGPROf<32>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<32, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x4 matrix in the first input by the 4x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1011101>];
}
#endif // VMFMAF3232X32X42BBF16_DEF

#ifndef VMFMAF3232X32X42BF16_DEF
#define VMFMAF3232X32X42BF16_DEF
def VMfmaF3232x32x42bF16 : VMfmaInst<"v_mfma_f32_32x32x4_2b_f16",
    /*dst0=*/[VGPROrAGPROf<32>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<32, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x4 matrix in the first input by the 4x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1001000>];
}
#endif // VMFMAF3232X32X42BF16_DEF

#ifndef VMFMAF3232X32X4XF32_DEF
#define VMFMAF3232X32X4XF32_DEF
def VMfmaF3232x32x4Xf32 : VMfmaInst<"v_mfma_f32_32x32x4_xf32",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let archs = [CDNA3Isa];
  let description = [{
    Multiply the 32x4 matrix in the first input by the 4x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3], 0b0111111>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA3], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAF3232X32X4XF32_DEF

#ifndef VMFMAF3232X32X64F8F6F4_DEF
#define VMFMAF3232X32X64F8F6F4_DEF
def VMfmaF3232x32x64F8f6f4 : VMfmaInst<"v_mfma_f32_32x32x64_f8f6f4",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<8>],
    /*src1=*/[VGPROrAGPROf<8>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Multiply the 32x64 matrix in the first input by the 64x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA4], 0b0101110>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAF3232X32X64F8F6F4_DEF

#ifndef VMFMAF3232X32X8BF16_DEF
#define VMFMAF3232X32X8BF16_DEF
def VMfmaF3232x32x8Bf16 : VMfmaInst<"v_mfma_f32_32x32x8_bf16",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x8 matrix in the first input by the 8x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1100000>];
}
#endif // VMFMAF3232X32X8BF16_DEF

#ifndef VMFMAF3232X32X8F16_DEF
#define VMFMAF3232X32X8F16_DEF
def VMfmaF3232x32x8F16 : VMfmaInst<"v_mfma_f32_32x32x8_f16",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x8 matrix in the first input by the 8x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1001100>];
}
#endif // VMFMAF3232X32X8F16_DEF

#ifndef VMFMAF324X4X116BF32_DEF
#define VMFMAF324X4X116BF32_DEF
def VMfmaF324x4x116bF32 : VMfmaInst<"v_mfma_f32_4x4x1_16b_f32",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<1>],
    /*src1=*/[VGPROrAGPROf<1>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 4x1 matrix in the first input by the 1x4 matrix in the second
    input and add the 4x4 matrix in the third input using fused multiply add.
    Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1000010>];
}
#endif // VMFMAF324X4X116BF32_DEF

#ifndef VMFMAF324X4X416BBF16_DEF
#define VMFMAF324X4X416BBF16_DEF
def VMfmaF324x4x416bBf16 : VMfmaInst<"v_mfma_f32_4x4x4_16b_bf16",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second
    input and add the 4x4 matrix in the third input using fused multiply add.
    Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1011111>];
}
#endif // VMFMAF324X4X416BBF16_DEF

#ifndef VMFMAF324X4X416BF16_DEF
#define VMFMAF324X4X416BF16_DEF
def VMfmaF324x4x416bF16 : VMfmaInst<"v_mfma_f32_4x4x4_16b_f16",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second
    input and add the 4x4 matrix in the third input using fused multiply add.
    Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1001010>];
}
#endif // VMFMAF324X4X416BF16_DEF

#ifndef VMFMAF6416X16X4F64_DEF
#define VMFMAF6416X16X4F64_DEF
def VMfmaF6416x16x4F64 : VMfmaInst<"v_mfma_f64_16x16x4_f64",
    /*dst0=*/[VGPROrAGPROf<8>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<8, InlineLitFP<[F64]>>]
  > {
  let description = [{
    Multiply the 16x4 matrix in the first input by the 4x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1101110>];
}
#endif // VMFMAF6416X16X4F64_DEF

#ifndef VMFMAF644X4X44BF64_DEF
#define VMFMAF644X4X44BF64_DEF
def VMfmaF644x4x44bF64 : VMfmaInst<"v_mfma_f64_4x4x4_4b_f64",
    /*dst0=*/[VGPROrAGPROf<2>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<2, InlineLitFP<[F64]>>]
  > {
  let description = [{
    Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second
    input and add the 4x4 matrix in the third input using fused multiply add.
    Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1101111>];
}
#endif // VMFMAF644X4X44BF64_DEF

#ifndef VMFMAI3216X16X32I8_DEF
#define VMFMAI3216X16X32I8_DEF
def VMfmaI3216x16x32I8 : VMfmaInst<"v_mfma_i32_16x16x32_i8",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitInt<[IntType<128>]>>]
  > {
  let description = [{
    Multiply the 16x32 matrix in the first input by the 32x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1010111>];
}
#endif // VMFMAI3216X16X32I8_DEF

#ifndef VMFMAI3216X16X44BI8_DEF
#define VMFMAI3216X16X44BI8_DEF
def VMfmaI3216x16x44bI8 : VMfmaInst<"v_mfma_i32_16x16x4_4b_i8",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<1>],
    /*src1=*/[VGPROrAGPROf<1>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitInt<[IntType<512>]>>]
  > {
  let description = [{
    Multiply the 16x4 matrix in the first input by the 4x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1010001>];
}
#endif // VMFMAI3216X16X44BI8_DEF

#ifndef VMFMAI3216X16X64I8_DEF
#define VMFMAI3216X16X64I8_DEF
def VMfmaI3216x16x64I8 : VMfmaInst<"v_mfma_i32_16x16x64_i8",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<4>],
    /*src1=*/[VGPROrAGPROf<4>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitInt<[IntType<128>]>>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Multiply the 16x64 matrix in the first input by the 64x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA4], 0b0110110>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAI3216X16X64I8_DEF

#ifndef VMFMAI3232X32X16I8_DEF
#define VMFMAI3232X32X16I8_DEF
def VMfmaI3232x32x16I8 : VMfmaInst<"v_mfma_i32_32x32x16_i8",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<2>],
    /*src1=*/[VGPROrAGPROf<2>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitInt<[IntType<512>]>>]
  > {
  let description = [{
    Multiply the 32x16 matrix in the first input by the 16x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1010110>];
}
#endif // VMFMAI3232X32X16I8_DEF

#ifndef VMFMAI3232X32X32I8_DEF
#define VMFMAI3232X32X32I8_DEF
def VMfmaI3232x32x32I8 : VMfmaInst<"v_mfma_i32_32x32x32_i8",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src0=*/[VGPROrAGPROf<4>],
    /*src1=*/[VGPROrAGPROf<4>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitInt<[IntType<512>]>>]
  > {
  let archs = [CDNA4Isa];
  let description = [{
    Multiply the 32x32 matrix in the first input by the 32x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA4], 0b0111000>];
  let asm_syntax = [
    ASMString<[VOP3P_MFMA_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2">
  ];
}
#endif // VMFMAI3232X32X32I8_DEF

#ifndef VMFMAI3232X32X42BI8_DEF
#define VMFMAI3232X32X42BI8_DEF
def VMfmaI3232x32x42bI8 : VMfmaInst<"v_mfma_i32_32x32x4_2b_i8",
    /*dst0=*/[VGPROrAGPROf<32>],
    /*src0=*/[VGPROrAGPROf<1>],
    /*src1=*/[VGPROrAGPROf<1>],
    /*src2=*/[VGPROrAGPROrLitOf<32, InlineLitInt<[IntType<1024>]>>]
  > {
  let description = [{
    Multiply the 32x4 matrix in the first input by the 4x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1010000>];
}
#endif // VMFMAI3232X32X42BI8_DEF

#ifndef VMFMAI324X4X416BI8_DEF
#define VMFMAI324X4X416BI8_DEF
def VMfmaI324x4x416bI8 : VMfmaInst<"v_mfma_i32_4x4x4_16b_i8",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src0=*/[VGPROrAGPROf<1>],
    /*src1=*/[VGPROrAGPROf<1>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitInt<[IntType<128>]>>]
  > {
  let description = [{
    Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second
    input and add the 4x4 matrix in the third input using fused multiply add.
    Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[VOP3P_MFMA_CDNA3, VOP3P_MFMA_CDNA4], 0b1010010>];
}
#endif // VMFMAI324X4X416BI8_DEF

#ifndef VMFMASCALEINST_DEF
#define VMFMASCALEINST_DEF
class VMfmaScaleInst<string mnemonic, list<Type> dst0Type, list<Type> src2Type> : AMDISAInstruction<mnemonic, [ScaledMMAOpInterface]> {
  let props = [IsVALU, IsVOP3P, IsMMA, IsScaledMMA];
  let archs = [CDNA4Isa];
  let outputs = (ins
    AnyRegOf<dst0Type>:$dst0
  );
  let inputs = (ins
    VGPROrAGPROf<8>:$src0,
    VGPROrAGPROf<8>:$src1,
    AnyOperandOf<src2Type>:$src2,
    OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>:$src3,
    OneGPROrSRegOrConstOf<InlineLitFP<[F32]>, InlineLitInt<[IntType<32>]>>:$src4
  );
  let trailingArgs = (ins
    CBSZAttr:$cbsz,
    ABIDAttr:$abid,
    BLGPAttr:$blgp,
    OpSelHiBase<3>:$op_sel_hi
  );
  let asm_syntax = [
    ASMString<[ENC_VOP3PX2_CDNA4], "${mnemonic} $dst0, $src0, $src1, $src2, $src3, $src4 $op_sel_hi$cbsz$abid$blgp">
  ];
}
#endif // VMFMASCALEINST_DEF

#ifndef VMFMASCALEF3216X16X128F8F6F4_DEF
#define VMFMASCALEF3216X16X128F8F6F4_DEF
def VMfmaScaleF3216x16x128F8f6f4 : VMfmaScaleInst<"v_mfma_scale_f32_16x16x128_f8f6f4",
    /*dst0=*/[VGPROrAGPROf<4>],
    /*src2=*/[VGPROrAGPROrLitOf<4, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 16x128 matrix in the first input by the 128x16 matrix in the
    second input and add the 16x16 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[ENC_VOP3PX2_CDNA4], 0b0101101>];
}
#endif // VMFMASCALEF3216X16X128F8F6F4_DEF

#ifndef VMFMASCALEF3232X32X64F8F6F4_DEF
#define VMFMASCALEF3232X32X64F8F6F4_DEF
def VMfmaScaleF3232x32x64F8f6f4 : VMfmaScaleInst<"v_mfma_scale_f32_32x32x64_f8f6f4",
    /*dst0=*/[VGPROrAGPROf<16>],
    /*src2=*/[VGPROrAGPROrLitOf<16, InlineLitFP<[F32]>>]
  > {
  let description = [{
    Multiply the 32x64 matrix in the first input by the 64x32 matrix in the
    second input and add the 32x32 matrix in the third input using fused
    multiply add. Store the resulting matrix into vector registers.
  }];
  let encodings = [InstEnc<[ENC_VOP3PX2_CDNA4], 0b0101110>];
}
#endif // VMFMASCALEF3232X32X64F8F6F4_DEF

#endif // AMDGCN_MMA_TD
