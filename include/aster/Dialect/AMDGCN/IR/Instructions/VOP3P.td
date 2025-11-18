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
// VOP3P Operation
//===----------------------------------------------------------------------===//

def VOP3PSrc : RegOrImm<[VGPRType, VGPRRangeType, AGPRType, AGPRRangeType], [I32, F32]>;
def VOP3PDst : AnyRegOf<[VGPRType, VGPRRangeType, AGPRType, AGPRRangeType]>;

def VOP3POp : AutoInstOp<"vop3p", [AMDGCNInstTraits]> {
  let summary = "VOP3P operation";
  let description = [{
    The VOP3P operation performs vector operations with three source operands.
  }];
  let inputs = (ins VOP3PSrc:$src0, OptionalReg<VOP3PSrc>:$src1, OptionalReg<VOP3PSrc>:$src2);
  let outputs = (ins VOP3PDst:$vdst);
  // TODO: add modifiers.
}

class VOP3PConstraints<int dst_size, int src0_size, int src1_size, int src2_size> {
  dag constraints = (ins
    /*ins=*/
      RegImmOperand<VOP3PSrc, src0_size>:$src0,
      !if(!gt(src1_size, -1), RegImmOperand<VOP3PSrc, src1_size>, NotPresent):$src1,
      !if(!gt(src2_size, -1), RegImmOperand<VOP3PSrc, src2_size>, NotPresent):$src2,
    /*outs=*/
      RegImmOperand<VOP3PDst, dst_size>:$vdst
  );
}

class VOP3PInst<string _mnemonic, list<Target> _targets>
    : AMDInst<_mnemonic, _mnemonic, VOP3POp, _targets> {
  // Default builders.
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$src0, CppValue:$src1, CppValue:$src2), [{
      return $_create($_builder, $_loc, $_opcode, vdst, src0, src1, src2);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$src0, PyValue:$src1, PyValue:$src2), [{
      return $_create(opcode=$_opcode, vdst=vdst, src0=src0, src1=src1, src2=src2, $_lastArgs)
    }]
  >;
  // Default ASM.
  let asmFormat = [SingleAsmVariant<"$vdst, $src0, $src1, $src2">];
}

//===----------------------------------------------------------------------===//
// Read/Write ACC Registers
//===----------------------------------------------------------------------===//

def VAccvgprRead : VOP3PInst<"v_accvgpr_read_b32",  [GFX942]> {
  let summary = "Move 32 bits of data from an ACC register to a VGPR";
  let constraints = (ins
    /*ins=*/RegImmOperand<AGPRType, 1>:$src0, NotPresent:$src1, NotPresent:$src2,
    /*outs=*/RegImmOperand<VGPRType, 1>:$vdst
  );
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$src0), [{
      return $_create($_builder, $_loc, $_opcode, vdst, src0, nullptr, nullptr);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$src0), [{
      return $_create(opcode=$_opcode, vdst=vdst, src0=src0, src1=None, src2=None, $_lastArgs)
    }]
  >;
  let asmFormat = [SingleAsmVariant<"$vdst, $src0">];
}

def VAccvgprWrite : VOP3PInst<"v_accvgpr_write_b32",  [GFX942]> {
  let summary = "Move 32 bits of data to an ACC register";
  let constraints = (ins
    /*ins=*/RegImmOperand<VOP3PSrc, 1>:$src0, NotPresent:$src1, NotPresent:$src2,
    /*outs=*/RegImmOperand<AGPRType, 1>:$vdst
  );
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$src0), [{
      return $_create($_builder, $_loc, $_opcode, vdst, src0, nullptr, nullptr);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$src0), [{
      return $_create(opcode=$_opcode, vdst=vdst, src0=src0, src1=None, src2=None, $_lastArgs)
    }]
  >;
  let asmFormat = [SingleAsmVariant<"$vdst, $src0">];
}

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
