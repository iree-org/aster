//===- WMMA.td - AMDGCN WMMA Instructions ------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_WMMA_TD
#define AMDGCN_WMMA_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/AMDGCN.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/InstOpInterfaces.td"

// Base class for gfx1250 WMMA instructions.
//
// Wave32 only.
//
// Operand indices must match MMAOpInterface:
//   odsIndex_dst0 = 0, odsIndex_src0 = 0, odsIndex_src1 = 1,
//   odsIndex_src2 = 2.
class WMMAInst<string mnemonic, int dstRegs, int src0Regs, int src1Regs,
               int src2Regs>
    : AMDISAInstruction<mnemonic, [MMAOpInterface]> {
  let props = [IsVALU, IsVOP3P, IsMMA, IsWMMA];
  let archs = [GFX12_50Isa];
  // No AGPR flavor.
  let outputs = (ins
    VGPROf<dstRegs>:$dst0
  );
  let inputs = (ins
    VGPROf<src0Regs>:$src0,
    VGPROf<src1Regs>:$src1,
    // C accumulator may be a VGPR or inline f32.
    AnyOperandOf<[VGPROf<src2Regs>, InlineLitFP<[F32]>]>:$src2
  );
  let trailingArgs = (ins
    MATRIX_A_REUSE:$matrix_a_reuse,
    MATRIX_B_REUSE:$matrix_b_reuse,
    NEGLOAttr:$neg_lo,
    NEGHIAttr:$neg_hi
  );
  let asm_syntax = [
    ASMString<[VOP3P_WMMA_GFX12_50], "${mnemonic} $dst0, $src0, $src1, $src2 [matrix_a_reuse] [matrix_b_reuse] $neg_lo$neg_hi">
  ];
}

// v_wmma_f32_16x16x32_f16: D[v8f32] = A[v16f16] * B[v16f16] + C[v8f32]
//   opcode 0x060 = 0b1100000
#ifndef VWMMAF3216X16X32F16_DEF
#define VWMMAF3216X16X32F16_DEF
def VWmmaF3216x16x32F16 : WMMAInst<"v_wmma_f32_16x16x32_f16",
    /*dstRegs=*/8, /*src0Regs=*/8, /*src1Regs=*/8, /*src2Regs=*/8> {
  let description = [{
    Wave32 only, the C accumulator may be a VGPR or inline f32.
  }];
  let encodings = [InstEnc<[VOP3P_WMMA_GFX12_50], 0b1100000>];
}
#endif // VWMMAF3216X16X32F16_DEF

// v_wmma_f32_16x16x32_bf16: D[v8f32] = A[v16bf16] * B[v16bf16] + C[v8f32]
//   opcode 0x062 = 0b1100010
#ifndef VWMMAF3216X16X32BF16_DEF
#define VWMMAF3216X16X32BF16_DEF
def VWmmaF3216x16x32Bf16 : WMMAInst<"v_wmma_f32_16x16x32_bf16",
    /*dstRegs=*/8, /*src0Regs=*/8, /*src1Regs=*/8, /*src2Regs=*/8> {
  let description = [{
    Wave32 only, the C accumulator may be a VGPR or inline f32.
  }];
  let encodings = [InstEnc<[VOP3P_WMMA_GFX12_50], 0b1100010>];
}
#endif // VWMMAF3216X16X32BF16_DEF

#endif // AMDGCN_WMMA_TD
