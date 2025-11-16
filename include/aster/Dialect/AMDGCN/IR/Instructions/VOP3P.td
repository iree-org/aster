//===- VOP3P.td - VOP3P Operations -------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines VOP3P specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_VOP3P_TD
#define AMDGCN_INST_VOP3P_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// VOP3P_MAI Operation
//===----------------------------------------------------------------------===//

def VOP3PMAIOp : VOP3P_Op<"vop3p_mai", [
    AMDGCNInstTraits,
    AMDGCNInstOpInterface,
    AllTypesMatch<["vdst", "result"]>,
    AllTypeIDsMatch<["vdst", "c"]>
  ]> {
  let summary = "VOP3P-MAI (Matrix Arithmetic Instructions) operation";
  let description = [{
    The VOP3P-MAI operation performs matrix arithmetic instructions for CDNA
    architecture. It takes three source operands (a, b, c) and a destination
    operand (vdst). Operands a and b are VGPR ranges, while c and vdst can be
    either VGPR ranges or AGPR ranges The result type
    matches the vdst type.

    VOP3P-MAI Format fields:
    - VDST [7:0]: Destination VGPR or AGPR
    - CBSZ [10:8]: Control Broadcast Size (0-4)
    - ABID [14:11]: A-matrix Broadcast Identifier (0-15)
    - ACC_CD [15]: Indicates that SRC-C and VDST use ACC VGPRs (AGPRs)
    - OP [22:16]: Opcode
    - ENCODING [31:24]: Must be: 110100111
    - A [40:32]: Source 0 operand (VGPR)
    - B [49:41]: Source 1 operand (VGPR)
    - C [58:50]: Source 2 operand (VGPR or AGPR)
    - ACC [60:59]: ACC[0] and ACC[1] flags
    - BLGP [63:61]: B-Matrix Lane-Group Pattern (0-7)
  }];
  let arguments = (ins
    AMDGCN_InstAttr:$opcode,
    VGPROrAGPRType:$vdst,
    VGPRRangeType:$a,
    VGPRRangeType:$b,
    VGPROrAGPRType:$c,
    CBSZAttr:$cbsz,
    ABIDAttr:$abid,
    BLGPAttr:$blgp
  );
  let results = (outs VGPROrAGPRType:$result);
  let assemblyFormat = [{
    $opcode $vdst `,` $a `,` $b `,` $c
    oilist(
        `cbsz` `=` $cbsz
      | `abid` `=` $abid
      | `blgp` `=` $blgp
    )
    attr-dict `:` type($a) `,` type($b) `,` type($c) `->` type($vdst)
  }];
  let extraClassDeclaration =[{
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getVdstMutable();
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().drop_front();
    }
  }];
}

class MFMAInst<string _mnemonic, string sym, int aRegs, int bRegs, int cRegs,
      int dstRegs, list<Target> _targets>
    : AMDInst<_mnemonic, sym, VOP3PMAIOp, _targets> {
  let constraints = (ins
      /*outs=*/AccOperand<dstRegs>:$vdst,
      /*ins=*/ VGPROperand<aRegs>:$a, VGPROperand<bRegs>:$b,
               AccOperand<cRegs>:$c);
  let asmFormat = [SingleAsmVariant<"$vdst, $a, $b, $c $cbsz$abid$blgp">];
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$vdst, PyValue:$a, PyValue:$b, PyValue:$c,
    IntArg<0>:$cbsz, IntArg<0>:$abid, IntArg<0>:$blgp
  )>;
}

def V_MFMA_F32_16X16X16_F16 : MFMAInst<
  /*mnemonic=*/"v_mfma_f32_16x16x16_f16", /*symbol=*/"v_mfma_f32_16x16x16_f16",
  /*aRegs=*/2, /*bRegs=*/2, /*cRegs=*/4, /*dstRegs=*/4, /*targets=*/[GFX942]> {
  let summary = "MFMA f32 16x16x16 with f16 inputs";
  let description = [{
    MFMA instruction performing matrix multiplication and accumulation
    with f32 output and f16 inputs.
  }];
}

def V_MFMA_F32_16X16X16_BF16 : MFMAInst<
  /*mnemonic=*/"v_mfma_f32_16x16x16_bf16", /*symbol=*/"v_mfma_f32_16x16x16_bf16",
  /*aRegs=*/2, /*bRegs=*/2, /*cRegs=*/4, /*dstRegs=*/4, /*targets=*/[GFX942]> {
  let summary = "MFMA f32 16x16x16 with bf16 inputs";
  let description = [{
    MFMA instruction performing matrix multiplication and accumulation
    with f32 output and bf16 inputs.
  }];
}

def V_MFMA_F16_16X16X16_F16 : MFMAInst<
  /*mnemonic=*/"v_mfma_f16_16x16x16_f16", /*symbol=*/"v_mfma_f16_16x16x16_f16",
  /*aRegs=*/2, /*bRegs=*/2, /*cRegs=*/4, /*dstRegs=*/4, /*targets=*/[GFX942]> {
  let summary = "MFMA f16 16x16x16 with f16 inputs";
  let description = [{
    MFMA instruction performing matrix multiplication and accumulation
    with f16 output and f16 inputs.
  }];
}

#endif // AMDGCN_INST_VOP3P_TD
