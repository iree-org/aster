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

def VOP3POp : AMDGCN_InstOp<"vop3p", [PureInst, InstInferType]> {
  let summary = "VOP3P operation";
  let description = [{
    The VOP3P operation performs vector operations with three source operands.
  }];
  let leadingArguments = (ins AMDGCN_InstAttr:$opcode);
  let inputs = (ins VOP3PSrc:$src0, OptionalReg<VOP3PSrc>:$src1, OptionalReg<VOP3PSrc>:$src2);
  let outputs = (ins VOP3PDst:$vdst);
  let assemblyFormat = [{
    custom<Opcode>($opcode) `outs` $vdst `ins` $src0
    (`src1` `=` $src1^)? (`src2` `=` $src2^)? attr-dict `:` type(operands)
  }];
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

class VOP3PInst<string _mnemonic, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, VOP3POp, _isa> {
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

def VAccvgprRead : VOP3PInst<"v_accvgpr_read_b32",  [CDNA3, CDNA4]> {
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

def VAccvgprWrite : VOP3PInst<"v_accvgpr_write_b32",  [CDNA3, CDNA4]> {
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

def VOP3PMAIOp : AMDGCN_InstOp<"vop3p.vop3p_mai", [
    PureInst, InstInferType,
    AllTypeIDsMatch<["vdst", "c"]>
  ]> {
  let summary = "VOP3P-MAI (Matrix Arithmetic Instructions) operation";
  let description = [{
    The VOP3P-MAI operation performs matrix arithmetic instructions for CDNA
    architecture. It takes three source operands (a, b, c) and a destination
    operand (vdst). Operands a and b are VGPR ranges, while c and vdst can be
    either VGPR ranges or AGPR ranges. The result type matches the vdst type.

    For scaled MFMA (f8f6f4 with ld_scale), use VOP3PScaledMAIOp instead.

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
  let leadingArguments = (ins AMDGCN_InstAttr:$opcode);
  let outputs = (ins VGPROrAGPRType:$vdst);
  let inputs = (ins VGPRRangeType:$a, VGPRRangeType:$b, VGPROrAGPRType:$c);
  let trailingArguments = (ins
    CBSZAttr:$cbsz,
    ABIDAttr:$abid,
    BLGPAttr:$blgp
  );
  let assemblyFormat = [{
    $opcode $vdst `,` $a `,` $b `,` $c
    oilist(
        `cbsz` `=` $cbsz
      | `abid` `=` $abid
      | `blgp` `=` $blgp
    )
    attr-dict `:` type($a) `,` type($b) `,` type($c) `->` type($vdst)
  }];
}

class MFMAInst<string _mnemonic, string sym, int aRegs, int bRegs, int cRegs,
      int dstRegs, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, sym, VOP3PMAIOp, _isa> {
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
  /*aRegs=*/2, /*bRegs=*/2, /*cRegs=*/4, /*dstRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "MFMA f32 16x16x16 with f16 inputs";
  let description = [{
    MFMA instruction performing matrix multiplication and accumulation
    with f32 output and f16 inputs.
  }];
}

def V_MFMA_F32_16X16X16_BF16 : MFMAInst<
  /*mnemonic=*/"v_mfma_f32_16x16x16_bf16", /*symbol=*/"v_mfma_f32_16x16x16_bf16",
  /*aRegs=*/2, /*bRegs=*/2, /*cRegs=*/4, /*dstRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "MFMA f32 16x16x16 with bf16 inputs";
  let description = [{
    MFMA instruction performing matrix multiplication and accumulation
    with f32 output and bf16 inputs.
  }];
}

def V_MFMA_F16_16X16X16_F16 : MFMAInst<
  /*mnemonic=*/"v_mfma_f16_16x16x16_f16", /*symbol=*/"v_mfma_f16_16x16x16_f16",
  /*aRegs=*/2, /*bRegs=*/2, /*cRegs=*/4, /*dstRegs=*/4, /*isa=*/[CDNA3, CDNA4]> {
  let summary = "MFMA f16 16x16x16 with f16 inputs";
  let description = [{
    MFMA instruction performing matrix multiplication and accumulation
    with f16 output and f16 inputs.
  }];
}

//===----------------------------------------------------------------------===//
// CDNA4 Scaled MFMA (VOP3PX encoding)
//===----------------------------------------------------------------------===//
// V_MFMA_SCALE_F32_* are single 4-DWORD instructions combining ld_scale +
// MFMA into one encoding (ISA manual section 7.2.1). In the IR we represent
// them as a single op with both MFMA operands (a, b, c/vdst) and scale
// source operands (scale_src0, scale_src1) plus format/op_sel modifiers.
//
// Format codes (cbsz for A, blgp for B):
//   0=fp8(E4M3), 1=bf8(E5M2), 2=fp6(E2M3), 3=bf6(E3M2), 4=fp4(E2M1)
//
// op_sel selects which byte of each scale source contains the E8M0 factor:
//   0: bits [7:0], 1: bits [15:8], 2: bits [23:16], 3: bits [31:24]
//
// ASM emission produces two lines (hardware assembler bundles them):
//   v_mfma_ld_scale_b32 <scale_src0>, <scale_src1> [op_sel_0] [op_sel_1]
//   v_mfma_f32_<tile>_f8f6f4 <vdst>, <a>, <b>, <c> [cbsz] [abid] [blgp]

def ScaleSrc : AnyRegOf<[VGPRType, SGPRType]>;

def VOP3PScaledMAIOp : AMDGCN_InstOp<"vop3p.vop3p_scaled_mai", [
    PureInst, InstInferType
  ]> {
  let summary = "VOP3P scaled MFMA (combined ld_scale + matrix multiply)";
  let description = [{
    Combined scaled MFMA instruction for CDNA4 (gfx950). This is a single
    4-DWORD VOP3PX encoding that bundles v_mfma_ld_scale_b32 with a
    v_mfma_f32_*_f8f6f4 MFMA instruction.

    Operands:
    - vdst/c: Destination/accumulator (VGPR or AGPR range)
    - a, b: Matrix input operands (VGPR ranges, always 8 regs)
    - scale_src0, scale_src1: Scale factor sources (VGPR or SGPR)

    Modifiers:
    - cbsz: A-matrix format code (0-4)
    - abid: A-matrix broadcast identifier (0-15)
    - blgp: B-matrix format/lane-group pattern (0-7)
    - op_sel_0, op_sel_1: Byte select for scale sources (0-3)
  }];
  let leadingArguments = (ins AMDGCN_InstAttr:$opcode);
  let outputs = (ins VGPROrAGPRType:$vdst);
  let inputs = (ins
    VGPRRangeType:$a,
    VGPRRangeType:$b,
    VGPROrAGPRType:$c,
    ScaleSrc:$scale_src0,
    ScaleSrc:$scale_src1
  );
  let trailingArguments = (ins
    CBSZAttr:$cbsz,
    ABIDAttr:$abid,
    BLGPAttr:$blgp,
    SquareIntModifier<I8Attr, "op_sel_0", 0>:$op_sel_0,
    SquareIntModifier<I8Attr, "op_sel_1", 0>:$op_sel_1
  );
  let assemblyFormat = [{
    $opcode $vdst `,` $a `,` $b `,` $c `,` $scale_src0 `,` $scale_src1
    oilist(
        `op_sel_0` `=` $op_sel_0
      | `op_sel_1` `=` $op_sel_1
      | `cbsz` `=` $cbsz
      | `abid` `=` $abid
      | `blgp` `=` $blgp
    )
    attr-dict `:` type($a) `,` type($b) `,` type($c) `,`
                  type($scale_src0) `,` type($scale_src1)
    `->` type($vdst)
  }];
}

class ScaledMFMAInst<string _mfma_mnemonic, string sym, int aRegs, int bRegs,
      int cRegs, int dstRegs, list<ISAVersion> _isa>
    : AMDInst<_mfma_mnemonic, sym, VOP3PScaledMAIOp, _isa> {
  let constraints = (ins
      /*outs=*/AccOperand<dstRegs>:$vdst,
      /*ins=*/ VGPROperand<aRegs>:$a, VGPROperand<bRegs>:$b,
               AccOperand<cRegs>:$c,
               RegImmOperand<ScaleSrc, 1>:$scale_src0,
               RegImmOperand<ScaleSrc, 1>:$scale_src1);
  // The generated ASM printer emits only the MFMA line. The ld_scale prefix
  // is emitted by custom code in TranslateModule.cpp::printInstruction().
  let asmFormat = [SingleAsmVariant<
    "$vdst, $a, $b, $c $cbsz$abid$blgp">];
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$vdst, PyValue:$a, PyValue:$b, PyValue:$c,
    PyValue:$scale_src0, PyValue:$scale_src1,
    IntArg<0>:$op_sel_0, IntArg<0>:$op_sel_1,
    IntArg<0>:$cbsz, IntArg<0>:$abid, IntArg<0>:$blgp
  )>;
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4 : ScaledMFMAInst<
  /*mfma_mnemonic=*/"v_mfma_f32_16x16x128_f8f6f4",
  /*symbol=*/"v_mfma_scale_f32_16x16x128_f8f6f4",
  /*aRegs=*/8, /*bRegs=*/8, /*cRegs=*/4, /*dstRegs=*/4, /*isa=*/[CDNA4]> {
  let summary = "Scaled MFMA f32 16x16x128 with f8/f6/f4 inputs";
  let description = [{
    Combined scaled MFMA: v_mfma_ld_scale_b32 + v_mfma_f32_16x16x128_f8f6f4
    in a single 4-DWORD VOP3PX encoding. Variable-precision narrow inputs
    (fp8/bf8/fp6/bf6/fp4) with per-block E8M0 exponent scaling.
  }];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4 : ScaledMFMAInst<
  /*mfma_mnemonic=*/"v_mfma_f32_32x32x64_f8f6f4",
  /*symbol=*/"v_mfma_scale_f32_32x32x64_f8f6f4",
  /*aRegs=*/8, /*bRegs=*/8, /*cRegs=*/16, /*dstRegs=*/16, /*isa=*/[CDNA4]> {
  let summary = "Scaled MFMA f32 32x32x64 with f8/f6/f4 inputs";
  let description = [{
    Combined scaled MFMA: v_mfma_ld_scale_b32 + v_mfma_f32_32x32x64_f8f6f4
    in a single 4-DWORD VOP3PX encoding. Variable-precision narrow inputs
    (fp8/bf8/fp6/bf6/fp4) with per-block E8M0 exponent scaling.
  }];
}

#endif // AMDGCN_INST_VOP3P_TD
