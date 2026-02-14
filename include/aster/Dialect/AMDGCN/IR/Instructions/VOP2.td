//===- VOP2.td - VOP2 Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines VOP2 specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_VOP2_TD
#define AMDGCN_INST_VOP2_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// VOP2 Operation
//===----------------------------------------------------------------------===//

def VOP2Src : RegOrImm<[SGPRType, VGPRType], [I32]>;
def VOP2Src1 : AnyRegOf<[VGPRType]>;
def VOP2Dst : AnyRegOf<[VGPRType]>;

def VOP2Op : AMDGCN_InstOp<"vop2",  [
    PureInst, InstInferType
  ]> {
  let summary = "VOP2 (Vector Operations with 2 operands) operation";
  let description = [{
    The VOP2 operation performs vector operations with two source operands.
  }];
  let leadingArguments = (ins AMDGCN_InstAttr:$opcode);
  let inputs = (ins VOP2Src:$src0, VOP2Src1:$src1, OptionalReg<SGPRType>:$src2);
  let outputs = (ins VOP2Dst:$vdst0, OptionalReg<SGPRType>:$dst1);
  // TODO: add declarative assembly format, currently it's impossible to do due
  // to a bug in MLIR tblgen when there are inferred types and op attr results.
  let hasCustomAssemblyFormat = 1;
  // TODO: add modifiers.
}

class VOP2Constraints<int sz, bit hasDst1, bit hasSrc2> {
  dag constraints = (ins
    /*ins=*/
      RegImmOperand<VOP2Src, sz>:$src0,
      RegImmOperand<VOP2Src1, sz>:$src1,
      !if(hasSrc2, RegImmOperand<SGPRType, 2>, NotPresent):$src2,
    /*outs=*/
      RegImmOperand<VOP2Dst, sz>:$vdst0,
      !if(hasDst1, RegImmOperand<SGPRType, 2>, NotPresent):$dst1
  );
}

class VOP2Inst<string _mnemonic, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, VOP2Op, _isa> {
  // Default constraints.
  let constraints = VOP2Constraints<1, /*dst1=*/0, /*src2=*/0>.constraints;
  // Default builders.
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$src0, CppValue:$src1), [{
      return $_create($_builder, $_loc, $_opcode, vdst, nullptr, src0, src1, nullptr);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$src0, PyValue:$src1), [{
      return $_create(opcode=$_opcode, vdst0=vdst, dst1=None, src0=src0, src1=src1, src2=None, $_lastArgs)
    }]
  >;
  // Default ASM.
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1">];
}

//===----------------------------------------------------------------------===//
// Bit operations
//===----------------------------------------------------------------------===//

def VLshrrevB32E32 : VOP2Inst<"v_lshrrev_b32_e32",  [CDNA3, CDNA4]> {
  let summary = "VOP2 logical shift right reverse 32-bit";
}

def VLshlrevB32E32 : VOP2Inst<"v_lshlrev_b32_e32", [CDNA3, CDNA4]> {
  let summary = "VOP2 logical shift left reverse 32-bit";
}

//===----------------------------------------------------------------------===//
// Conditional and mask operations
//===----------------------------------------------------------------------===//

def VCndmaskB32 : VOP2Inst<"v_cndmask_b32", [CDNA3, CDNA4]> {
  let summary = "VOP2 conditional mask";
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1, $src2">];
  let constraints = VOP2Constraints<1, /*dst1=*/0, /*src2=*/1>.constraints;
  // Default builders.
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$src0, CppValue:$src1, CppValue:$src2), [{
      return $_create($_builder, $_loc, $_opcode, vdst, nullptr, src0, src1, src2);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$src0, PyValue:$src1, PyValue:$src2), [{
      return $_create(opcode=$_opcode, vdst0=vdst, dst1=None, src0=src0, src1=src1, src2=src2, $_lastArgs)
    }]
  >;
}

//===----------------------------------------------------------------------===//
// Floating-point arithmetic operations
//===----------------------------------------------------------------------===//

def VAddF32 : VOP2Inst<"v_add_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 add two 32-bit floating point values";
}

def VSubF32 : VOP2Inst<"v_sub_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 subtract 32-bit floating point values";
}

def VSubrevF32 : VOP2Inst<"v_subrev_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 reverse subtract 32-bit floating point values";
}

def VFmacF64 : VOP2Inst<"v_fmac_f64", [CDNA3, CDNA4]> {
  let summary = "VOP2 fused multiply-add 64-bit floating point";
  let constraints = VOP2Constraints<2, /*dst1=*/0, /*src2=*/1>.constraints;
}

def VMulF32 : VOP2Inst<"v_mul_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 multiply 32-bit floating point values";
}

def VMinF32 : VOP2Inst<"v_min_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 minimum of two 32-bit floating point values";
}

def VMaxF32 : VOP2Inst<"v_max_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 maximum of two 32-bit floating point values";
}

//===----------------------------------------------------------------------===//
// 16-bit floating-point arithmetic operations
//===----------------------------------------------------------------------===//

def VAddF16 : VOP2Inst<"v_add_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 add two 16-bit floating point values";
}

def VSubF16 : VOP2Inst<"v_sub_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 subtract 16-bit floating point values";
}

def VSubrevF16 : VOP2Inst<"v_subrev_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 reverse subtract 16-bit floating point values";
}

def VMulF16 : VOP2Inst<"v_mul_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 multiply 16-bit floating point values";
}

def VMacF16 : VOP2Inst<"v_mac_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 multiply-accumulate 16-bit floating point";
}

def VMadmkF16 : VOP2Inst<"v_madmk_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 MAD with literal constant multiplier (16-bit)";
}

def VMadakF16 : VOP2Inst<"v_madak_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 MAD with literal constant addend (16-bit)";
}

def VMaxF16 : VOP2Inst<"v_max_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 maximum of two 16-bit floating point values";
}

def VMinF16 : VOP2Inst<"v_min_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 minimum of two 16-bit floating point values";
}

def VLdexpF16 : VOP2Inst<"v_ldexp_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 ldexp 16-bit floating point";
}

//===----------------------------------------------------------------------===//
// 16-bit integer arithmetic operations
//===----------------------------------------------------------------------===//

def VAddU16 : VOP2Inst<"v_add_u16", [CDNA3, CDNA4]> {
  let summary = "VOP2 add two unsigned 16-bit integers";
}

def VSubU16 : VOP2Inst<"v_sub_u16", [CDNA3, CDNA4]> {
  let summary = "VOP2 subtract unsigned 16-bit integers";
}

def VSubrevU16 : VOP2Inst<"v_subrev_u16", [CDNA3, CDNA4]> {
  let summary = "VOP2 reverse subtract unsigned 16-bit integers";
}

def VMulLoU16 : VOP2Inst<"v_mul_lo_u16", [CDNA3, CDNA4]> {
  let summary = "VOP2 multiply unsigned 16-bit integers (low bits)";
}

//===----------------------------------------------------------------------===//
// 16-bit shift operations
//===----------------------------------------------------------------------===//

def VLshlrevB16 : VOP2Inst<"v_lshlrev_b16", [CDNA3, CDNA4]> {
  let summary = "VOP2 logical shift left reverse 16-bit";
}

def VLshrrevB16 : VOP2Inst<"v_lshrrev_b16", [CDNA3, CDNA4]> {
  let summary = "VOP2 logical shift right reverse 16-bit";
}

def VAshrrevI16 : VOP2Inst<"v_ashrrev_i16", [CDNA3, CDNA4]> {
  let summary = "VOP2 arithmetic shift right reverse 16-bit";
}

//===----------------------------------------------------------------------===//
// 16-bit integer min/max operations
//===----------------------------------------------------------------------===//

def VMaxU16 : VOP2Inst<"v_max_u16", [CDNA3, CDNA4]> {
  let summary = "VOP2 maximum of two unsigned 16-bit integers";
}

def VMaxI16 : VOP2Inst<"v_max_i16", [CDNA3, CDNA4]> {
  let summary = "VOP2 maximum of two signed 16-bit integers";
}

def VMinU16 : VOP2Inst<"v_min_u16", [CDNA3, CDNA4]> {
  let summary = "VOP2 minimum of two unsigned 16-bit integers";
}

def VMinI16 : VOP2Inst<"v_min_i16", [CDNA3, CDNA4]> {
  let summary = "VOP2 minimum of two signed 16-bit integers";
}

//===----------------------------------------------------------------------===//
// Integer min/max operations
//===----------------------------------------------------------------------===//

def VMinI32 : VOP2Inst<"v_min_i32", [CDNA3, CDNA4]> {
  let summary = "VOP2 minimum of two signed 32-bit integers";
}

def VMaxI32 : VOP2Inst<"v_max_i32", [CDNA3, CDNA4]> {
  let summary = "VOP2 maximum of two signed 32-bit integers";
}

def VMinU32 : VOP2Inst<"v_min_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 minimum of two unsigned 32-bit integers";
}

def VMaxU32 : VOP2Inst<"v_max_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 maximum of two unsigned 32-bit integers";
}

//===----------------------------------------------------------------------===//
// Bit operations
//===----------------------------------------------------------------------===//

def VLshrrevB32 : VOP2Inst<"v_lshrrev_b32", [CDNA3, CDNA4]> {
  let summary = "VOP2 logical shift right reverse 32-bit";
}

def VAshrrevI32 : VOP2Inst<"v_ashrrev_i32", [CDNA3, CDNA4]> {
  let summary = "VOP2 arithmetic shift right reverse 32-bit";
}

def VAndB32 : VOP2Inst<"v_and_b32", [CDNA3, CDNA4]> {
  let summary = "VOP2 bitwise AND";
}

def VOrB32 : VOP2Inst<"v_or_b32", [CDNA3, CDNA4]> {
  let summary = "VOP2 bitwise OR";
}

def VXorB32 : VOP2Inst<"v_xor_b32", [CDNA3, CDNA4]> {
  let summary = "VOP2 bitwise XOR";
}

//===----------------------------------------------------------------------===//
// Floating-point FMA with constant operations
//===----------------------------------------------------------------------===//

def VFmamkF32 : VOP2Inst<"v_fmamk_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 FMA with literal constant multiplier";
}

def VFmaakF32 : VOP2Inst<"v_fmaak_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 FMA with literal constant addend";
}

//===----------------------------------------------------------------------===//
// Integer multiply operations
//===----------------------------------------------------------------------===//

def VMulI32I24 : VOP2Inst<"v_mul_i32_i24", [CDNA3, CDNA4]> {
  let summary = "VOP2 multiply signed 24-bit integers";
}

def VMulHiI32I24 : VOP2Inst<"v_mul_hi_i32_i24", [CDNA3, CDNA4]> {
  let summary = "VOP2 multiply signed 24-bit integers (high 32 bits)";
}

def VMulU32U24 : VOP2Inst<"v_mul_u32_u24", [CDNA3, CDNA4]> {
  let summary = "VOP2 multiply unsigned 24-bit integers";
}

def VMulHiU32U24 : VOP2Inst<"v_mul_hi_u32_u24", [CDNA3, CDNA4]> {
  let summary = "VOP2 multiply unsigned 24-bit integers (high 32 bits)";
}

//===----------------------------------------------------------------------===//
// Add operations
//===----------------------------------------------------------------------===//

def VAddCoU32 : VOP2Inst<"v_add_co_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 add with carry out";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1">];
  let constraints = VOP2Constraints<1, /*dst1=*/1, /*src2=*/0>.constraints;
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$dst1, CppValue:$src0, CppValue:$src1), [{
      return $_create($_builder, $_loc, $_opcode, vdst, dst1, src0, src1, nullptr);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$dst1, PyValue:$src0, PyValue:$src1), [{
      return $_create(opcode=$_opcode, vdst0=vdst, dst1=dst1, src0=src0, src1=src1, src2=None, $_lastArgs)
    }]
  >;
}

//===----------------------------------------------------------------------===//
// Subtract operations
//===----------------------------------------------------------------------===//

def VSubCoU32 : VOP2Inst<"v_sub_co_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 subtract with carry out";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1">];
  let constraints = VOP2Constraints<1, /*dst1=*/1, /*src2=*/0>.constraints;
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$dst1, CppValue:$src0, CppValue:$src1), [{
      return $_create($_builder, $_loc, $_opcode, vdst, dst1, src0, src1, nullptr);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$dst1, PyValue:$src0, PyValue:$src1), [{
      return $_create(opcode=$_opcode, vdst0=vdst, dst1=dst1, src0=src0, src1=src1, src2=None, $_lastArgs)
    }]
  >;
}

def VSubrevCoU32 : VOP2Inst<"v_subrev_co_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 reverse subtract with carry out";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1">];
  let constraints = VOP2Constraints<1, /*dst1=*/1, /*src2=*/0>.constraints;
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$dst1, CppValue:$src0, CppValue:$src1), [{
      return $_create($_builder, $_loc, $_opcode, vdst, dst1, src0, src1, nullptr);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$dst1, PyValue:$src0, PyValue:$src1), [{
      return $_create(opcode=$_opcode, vdst0=vdst, dst1=dst1, src0=src0, src1=src1, src2=None, $_lastArgs)
    }]
  >;
}

def VAddcCoU32 : VOP2Inst<"v_addc_co_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 add with carry in and out";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1, $src2">];
  let constraints = VOP2Constraints<1, /*dst1=*/1, /*src2=*/1>.constraints;
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$dst1, CppValue:$src0, CppValue:$src1, CppValue:$src2), [{
      return $_create($_builder, $_loc, $_opcode, vdst, dst1, src0, src1, src2);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$dst1, PyValue:$src0, PyValue:$src1, PyValue:$src2), [{
      return $_create(opcode=$_opcode, vdst0=vdst, dst1=dst1, src0=src0, src1=src1, src2=src2, $_lastArgs)
    }]
  >;
}

def VSubbCoU32 : VOP2Inst<"v_subb_co_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 subtract with borrow (carry in and out)";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1, $src2">];
  let constraints = VOP2Constraints<1, /*dst1=*/1, /*src2=*/1>.constraints;
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$dst1, CppValue:$src0, CppValue:$src1, CppValue:$src2), [{
      return $_create($_builder, $_loc, $_opcode, vdst, dst1, src0, src1, src2);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$dst1, PyValue:$src0, PyValue:$src1, PyValue:$src2), [{
      return $_create(opcode=$_opcode, vdst0=vdst, dst1=dst1, src0=src0, src1=src1, src2=src2, $_lastArgs)
    }]
  >;
}

def VSubbrevCoU32 : VOP2Inst<"v_subbrev_co_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 reverse subtract with borrow (carry in and out)";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1, $src2">];
  let constraints = VOP2Constraints<1, /*dst1=*/1, /*src2=*/1>.constraints;
  let cppBuilder = OpBuilder<(ins
    CppValue:$vdst, CppValue:$dst1, CppValue:$src0, CppValue:$src1, CppValue:$src2), [{
      return $_create($_builder, $_loc, $_opcode, vdst, dst1, src0, src1, src2);
    }]
  >;
  let pythonBuilder = OpBuilder<(ins
    PyValue:$vdst, PyValue:$dst1, PyValue:$src0, PyValue:$src1, PyValue:$src2), [{
      return $_create(opcode=$_opcode, vdst0=vdst, dst1=dst1, src0=src0, src1=src1, src2=src2, $_lastArgs)
    }]
  >;
}

def VAddU32 : VOP2Inst<"v_add_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 add two unsigned 32-bit integers";
}

def VSubU32 : VOP2Inst<"v_sub_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 subtract unsigned 32-bit integers";
}

def VSubrevU32 : VOP2Inst<"v_subrev_u32", [CDNA3, CDNA4]> {
  let summary = "VOP2 reverse subtract unsigned 32-bit integers";
}

//===----------------------------------------------------------------------===//
// Dot product operations
//===----------------------------------------------------------------------===//

def VDot2cF32F16 : VOP2Inst<"v_dot2c_f32_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 2-element f16 dot product with f32 accumulation";
}

def VDot2cI32I16 : VOP2Inst<"v_dot2c_i32_i16", [CDNA3, CDNA4]> {
  let summary = "VOP2 2-element i16 dot product with i32 accumulation";
}

def VDot4cI32I8 : VOP2Inst<"v_dot4c_i32_i8", [CDNA3, CDNA4]> {
  let summary = "VOP2 4-element i8 dot product with i32 accumulation";
}

def VDot8cI32I4 : VOP2Inst<"v_dot8c_i32_i4", [CDNA3, CDNA4]> {
  let summary = "VOP2 8-element i4 dot product with i32 accumulation";
}

def VFmacF32 : VOP2Inst<"v_fmac_f32", [CDNA3, CDNA4]> {
  let summary = "VOP2 fused multiply-accumulate 32-bit floating point";
}

def VPkFmacF16 : VOP2Inst<"v_pk_fmac_f16", [CDNA3, CDNA4]> {
  let summary = "VOP2 packed fused multiply-accumulate 16-bit floating point";
}

def VXnorB32 : VOP2Inst<"v_xnor_b32", [CDNA3, CDNA4]> {
  let summary = "VOP2 bitwise XNOR";
}

def VAddI16 : VOP2Inst<"v_add_i16", [CDNA3, CDNA4]> {
  let summary = "VOP2 add on 16-bit signed operands";
}

def VAddI32 : VOP2Inst<"v_add_i32", [CDNA3, CDNA4]> {
  let summary = "VOP2 add on 32-bit signed operands";
}

#endif // AMDGCN_INST_VOP2_TD
