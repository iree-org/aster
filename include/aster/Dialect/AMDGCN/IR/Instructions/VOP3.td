//===- VOP3.td - VOP3 Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines VOP3 specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_VOP3_TD
#define AMDGCN_INST_VOP3_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"


//===----------------------------------------------------------------------===//
// VOP3 Operation
//===----------------------------------------------------------------------===//

def VOP3Src : RegOrImm<[SGPRType, VGPRType, SGPRRangeType, VGPRRangeType], [I32]>;
def VOP3Dst : AnyRegOf<[VGPRType, VGPRRangeType]>;

def VOP3Op : AutoInstOp<"vop3", [AMDGCNInstTraits]> {
  let summary = "VOP3 (Vector Operations with 3 operands) operation";
  let description = [{
    The VOP3 operation performs vector operations with three source operands.
  }];
  let inputs = (ins VOP3Src:$src0, VOP3Src:$src1, OptionalReg<VOP3Src>:$src2);
  let outputs = (ins VOP3Dst:$vdst0, OptionalReg<SGPRRangeType>:$dst1);
  // TODO: add modifiers.
}

class VOP3Constraints<int words, int dst1_w, int src2_w> {
  dag constraints = (ins
    /*ins=*/
      RegImmOperand<VOP3Src, words>:$src0,
      RegImmOperand<VOP3Src, words>:$src1,
      !if(!gt(src2_w, 0), RegImmOperand<VOP3Src, src2_w>, NotPresent):$src2,
    /*outs=*/
      RegImmOperand<VOP3Dst, words>:$vdst0,
      !if(!gt(dst1_w, 0), RegImmOperand<SGPRRangeType, dst1_w>, NotPresent):$dst1
  );
}

class VOP3Inst<string _mnemonic, list<Target> _targets>
    : AMDInst<_mnemonic, _mnemonic, VOP3Op, _targets> {
  // Default constraints.
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/0, /*src2_w=*/0>.constraints;
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
// Conditional and mask operations
//===----------------------------------------------------------------------===//

def VCndmaskB32E64 : VOP3Inst<"v_cndmask_b32_e64", [GFX942]> {
  let summary = "VOP3 conditional mask";
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1, $src2">];
  let constraints = (ins
    /*outs=*/NotPresent:$dst1,
    /*ins=*/ Present:$src2
  );
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

def VAddF32E64 : VOP3Inst<"v_add_f32_e64", [GFX942]> {
  let summary = "VOP3 add two 32-bit floating point values";
}

def VSubF32E64 : VOP3Inst<"v_sub_f32_e64", [GFX942]> {
  let summary = "VOP3 subtract 32-bit floating point values";
}

def VSubrevF32E64 : VOP3Inst<"v_subrev_f32_e64", [GFX942]> {
  let summary = "VOP3 reverse subtract 32-bit floating point values";
}

def VFmacF64E64 : VOP3Inst<"v_fmac_f64_e64", [GFX942]> {
  let summary = "VOP3 fused multiply-add 64-bit floating point";
  let constraints = VOP3Constraints</*words=*/2, /*dst1_w=*/0, /*src2_w=*/0>.constraints;
}

def VMulF32E64 : VOP3Inst<"v_mul_f32_e64", [GFX942]> {
  let summary = "VOP3 multiply 32-bit floating point values";
}

def VMinF32E64 : VOP3Inst<"v_min_f32_e64", [GFX942]> {
  let summary = "VOP3 minimum of two 32-bit floating point values";
}

def VMaxF32E64 : VOP3Inst<"v_max_f32_e64", [GFX942]> {
  let summary = "VOP3 maximum of two 32-bit floating point values";
}

//===----------------------------------------------------------------------===//
// 16-bit floating-point arithmetic operations
//===----------------------------------------------------------------------===//

def VAddF16E64 : VOP3Inst<"v_add_f16_e64", [GFX942]> {
  let summary = "VOP3 add two 16-bit floating point values";
}

def VSubF16E64 : VOP3Inst<"v_sub_f16_e64", [GFX942]> {
  let summary = "VOP3 subtract 16-bit floating point values";
}

def VSubrevF16E64 : VOP3Inst<"v_subrev_f16_e64", [GFX942]> {
  let summary = "VOP3 reverse subtract 16-bit floating point values";
}

def VMulF16E64 : VOP3Inst<"v_mul_f16_e64", [GFX942]> {
  let summary = "VOP3 multiply 16-bit floating point values";
}

def VMacF16E64 : VOP3Inst<"v_mac_f16_e64", [GFX942]> {
  let summary = "VOP3 multiply-accumulate 16-bit floating point";
}

def VMaxF16E64 : VOP3Inst<"v_max_f16_e64", [GFX942]> {
  let summary = "VOP3 maximum of two 16-bit floating point values";
}

def VMinF16E64 : VOP3Inst<"v_min_f16_e64", [GFX942]> {
  let summary = "VOP3 minimum of two 16-bit floating point values";
}

def VLdexpF16E64 : VOP3Inst<"v_ldexp_f16_e64", [GFX942]> {
  let summary = "VOP3 ldexp 16-bit floating point";
}

//===----------------------------------------------------------------------===//
// 16-bit integer arithmetic operations
//===----------------------------------------------------------------------===//

def VAddU16E64 : VOP3Inst<"v_add_u16_e64", [GFX942]> {
  let summary = "VOP3 add two unsigned 16-bit integers";
}

def VSubU16E64 : VOP3Inst<"v_sub_u16_e64", [GFX942]> {
  let summary = "VOP3 subtract unsigned 16-bit integers";
}

def VSubrevU16E64 : VOP3Inst<"v_subrev_u16_e64", [GFX942]> {
  let summary = "VOP3 reverse subtract unsigned 16-bit integers";
}

def VMulLoU16E64 : VOP3Inst<"v_mul_lo_u16_e64", [GFX942]> {
  let summary = "VOP3 multiply unsigned 16-bit integers (low bits)";
}

//===----------------------------------------------------------------------===//
// 16-bit shift operations
//===----------------------------------------------------------------------===//

def VLshlrevB16E64 : VOP3Inst<"v_lshlrev_b16_e64", [GFX942]> {
  let summary = "VOP3 logical shift left reverse 16-bit";
}

def VLshrrevB16E64 : VOP3Inst<"v_lshrrev_b16_e64", [GFX942]> {
  let summary = "VOP3 logical shift right reverse 16-bit";
}

def VAshrrevI16E64 : VOP3Inst<"v_ashrrev_i16_e64", [GFX942]> {
  let summary = "VOP3 arithmetic shift right reverse 16-bit";
}

//===----------------------------------------------------------------------===//
// 16-bit integer min/max operations
//===----------------------------------------------------------------------===//

def VMaxU16E64 : VOP3Inst<"v_max_u16_e64", [GFX942]> {
  let summary = "VOP3 maximum of two unsigned 16-bit integers";
}

def VMaxI16E64 : VOP3Inst<"v_max_i16_e64", [GFX942]> {
  let summary = "VOP3 maximum of two signed 16-bit integers";
}

def VMinU16E64 : VOP3Inst<"v_min_u16_e64", [GFX942]> {
  let summary = "VOP3 minimum of two unsigned 16-bit integers";
}

def VMinI16E64 : VOP3Inst<"v_min_i16_e64", [GFX942]> {
  let summary = "VOP3 minimum of two signed 16-bit integers";
}

//===----------------------------------------------------------------------===//
// Integer min/max operations
//===----------------------------------------------------------------------===//

def VMinI32E64 : VOP3Inst<"v_min_i32_e64", [GFX942]> {
  let summary = "VOP3 minimum of two signed 32-bit integers";
}

def VMaxI32E64 : VOP3Inst<"v_max_i32_e64", [GFX942]> {
  let summary = "VOP3 maximum of two signed 32-bit integers";
}

def VMinU32E64 : VOP3Inst<"v_min_u32_e64", [GFX942]> {
  let summary = "VOP3 minimum of two unsigned 32-bit integers";
}

def VMaxU32E64 : VOP3Inst<"v_max_u32_e64", [GFX942]> {
  let summary = "VOP3 maximum of two unsigned 32-bit integers";
}

//===----------------------------------------------------------------------===//
// Bit operations
//===----------------------------------------------------------------------===//

def VLshrrevB32E64 : VOP3Inst<"v_lshrrev_b32_e64", [GFX942]> {
  let summary = "VOP3 logical shift right 32-bit";
}

def VLshlrevB32E64 : VOP3Inst<"v_lshlrev_b32_e64", [GFX942]> {
  let summary = "VOP3 logical shift left 32-bit";
}

def VLshlrevB64 : VOP3Inst<"v_lshlrev_b64", [GFX942]> {
  let summary = "VOP3 logical shift left 64-bit";
  let constraints = VOP3Constraints</*words=*/2, /*dst1_w=*/0, /*src2_w=*/0>.constraints;
}

def VLshrrevB64 : VOP3Inst<"v_lshrrev_b64", [GFX942]> {
  let summary = "VOP3 logical shift right 64-bit";
  let constraints = VOP3Constraints</*words=*/2, /*dst1_w=*/0, /*src2_w=*/0>.constraints;
}

def VAshrrevI32E64 : VOP3Inst<"v_ashrrev_i32_e64", [GFX942]> {
  let summary = "VOP3 arithmetic shift right 32-bit";
}

def VAshrrevI64 : VOP3Inst<"v_ashrrev_i64", [GFX942]> {
  let summary = "VOP3 arithmetic shift right 64-bit";
  let constraints = VOP3Constraints</*words=*/2, /*dst1_w=*/0, /*src2_w=*/0>.constraints;
}

def VAndB32E64 : VOP3Inst<"v_and_b32_e64", [GFX942]> {
  let summary = "VOP3 bitwise AND";
}

def VOrB32E64 : VOP3Inst<"v_or_b32_e64", [GFX942]> {
  let summary = "VOP3 bitwise OR";
}

def VXorB32E64 : VOP3Inst<"v_xor_b32_e64", [GFX942]> {
  let summary = "VOP3 bitwise XOR";
}

//===----------------------------------------------------------------------===//
// Integer multiply operations
//===----------------------------------------------------------------------===//

defvar VMad64Constraint = (ins
    /*ins=*/
      RegImmOperand<VOP3Src, 1>:$src0,
      RegImmOperand<VOP3Src, 1>:$src1,
      RegImmOperand<VOP3Src, 2>:$src2,
    /*outs=*/
      RegImmOperand<VOP3Dst, 2>:$vdst0,
      RegImmOperand<SGPRRangeType, 2>:$dst1
  );

def VMadU64U32 : VOP3Inst<"v_mad_u64_u32", [GFX942]> {
  let summary = "VOP3 multiply-add unsigned 32-bit to 64-bit with carry";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1, $src2">];
  let constraints = VMad64Constraint;
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

def VMadI64I32 : VOP3Inst<"v_mad_i64_i32", [GFX942]> {
  let summary = "VOP3 multiply-add signed 32-bit to 64-bit with carry";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1, $src2">];
  let constraints = VMad64Constraint;
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

def VMadU32U16 : VOP3Inst<"v_mad_u32_u16", [GFX942]> {
  let summary = "VOP3 multiply-add unsigned 16-bit to 32-bit";
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1, $src2">];
  let constraints = (ins
    /*outs=*/NotPresent:$dst1,
    /*ins=*/ Present:$src2
  );
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/0, /*src2_w=*/1>.constraints;
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

def VMadI32I16 : VOP3Inst<"v_mad_i32_i16", [GFX942]> {
  let summary = "VOP3 multiply-add signed 16-bit to 32-bit";
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1, $src2">];
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/0, /*src2_w=*/1>.constraints;
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

def VMulI32I24E64 : VOP3Inst<"v_mul_i32_i24_e64", [GFX942]> {
  let summary = "VOP3 multiply signed 24-bit integers";
}

def VMulHiI32I24E64 : VOP3Inst<"v_mul_hi_i32_i24_e64", [GFX942]> {
  let summary = "VOP3 multiply signed 24-bit integers (high 32 bits)";
}

def VMulU32U24E64 : VOP3Inst<"v_mul_u32_u24_e64", [GFX942]> {
  let summary = "VOP3 multiply unsigned 24-bit integers";
}

def VMulHiU32U24E64 : VOP3Inst<"v_mul_hi_u32_u24_e64", [GFX942]> {
  let summary = "VOP3 multiply unsigned 24-bit integers (high 32 bits)";
}

def VMulLoU32 : VOP3Inst<"v_mul_lo_u32", [GFX942]> {
  let summary = "VOP3 mul lo 32-bit integers";
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1">];
}

def VMulHiU32 : VOP3Inst<"v_mul_hi_u32", [GFX942]> {
  let summary = "VOP3 mul hi 32-bit integers";
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1">];
}

def VMulHiI32 : VOP3Inst<"v_mul_hi_i32", [GFX942]> {
  let summary = "VOP3 mul hi 32-bit signed integers";
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1">];
}

//===----------------------------------------------------------------------===//
// Add operations
//===----------------------------------------------------------------------===//

def VAddCoU32E64 : VOP3Inst<"v_add_co_u32_e64", [GFX942]> {
  let summary = "VOP3 add with carry out";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1">];
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/2, /*src2_w=*/0>.constraints;
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

def VAdd3U32 : VOP3Inst<"v_add3_u32", [GFX942]> {
  let summary = "VOP3 add with three 32-bit operands";
  let asmFormat = [SingleAsmVariant<"$vdst0, $src0, $src1, $src2">];
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/0, /*src2_w=*/1>.constraints;
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
// Subtract operations
//===----------------------------------------------------------------------===//

def VSubCoU32E64 : VOP3Inst<"v_sub_co_u32_e64", [GFX942]> {
  let summary = "VOP3 subtract with carry out";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1">];
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/2, /*src2_w=*/0>.constraints;
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

def VSubrevCoU32E64 : VOP3Inst<"v_subrev_co_u32_e64", [GFX942]> {
  let summary = "VOP3 reverse subtract with carry out";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1">];
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/2, /*src2_w=*/0>.constraints;
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

def VAddcCoU32E64 : VOP3Inst<"v_addc_co_u32_e64", [GFX942]> {
  let summary = "VOP3 add with carry in and out";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1, $src2">];
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/1, /*src2_w=*/1>.constraints;
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

def VSubbCoU32E64 : VOP3Inst<"v_subb_co_u32_e64", [GFX942]> {
  let summary = "VOP3 subtract with borrow (carry in and out)";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1, $src2">];
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/2, /*src2_w=*/2>.constraints;
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

def VSubbrevCoU32E64 : VOP3Inst<"v_subbrev_co_u32_e64", [GFX942]> {
  let summary = "VOP3 reverse subtract with borrow (carry in and out)";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1, $src2">];
  let constraints = VOP3Constraints</*words=*/1, /*dst1_w=*/2, /*src2_w=*/2>.constraints;
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

def VAddU32E64 : VOP3Inst<"v_add_u32_e64", [GFX942]> {
  let summary = "VOP3 add two unsigned 32-bit integers";
}

def VSubU32E64 : VOP3Inst<"v_sub_u32_e64", [GFX942]> {
  let summary = "VOP3 subtract unsigned 32-bit integers";
}

def VSubrevU32E64 : VOP3Inst<"v_subrev_u32_e64", [GFX942]> {
  let summary = "VOP3 reverse subtract unsigned 32-bit integers";
}

def VLshlAddU64 : VOP3Inst<"v_lshl_add_u64", [GFX942]> {
  let summary = "VOP3 lshl add 64-bit integers";
  let asmFormat = [SingleAsmVariant<"$vdst0, $dst1, $src0, $src1">];
  let constraints = VOP3Constraints</*words=*/2, /*dst1_w=*/0, /*src2_w=*/2>.constraints;
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

#endif // AMDGCN_INST_VOP3_TD
