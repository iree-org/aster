//===- SOP2.td - SOP2 Operations ---------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SOP2 specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_SOP2_TD
#define AMDGCN_INST_SOP2_TD

include "aster/Dialect/AMDGCN/IR/AMDGCN.td"

//===----------------------------------------------------------------------===//
// SOP2Op Operation
//===----------------------------------------------------------------------===//

def SOP2SSrc : RegOrImm<[SGPRType, SGPRRangeType], [I32]>;

def SOP2Op : AutoInstOp<"sop2", [AMDGCNInstTraits]> {
  let summary = "SOP2 (Scalar Operations with 2 operands) operation";
  let description = [{
    The SOP2 operation performs scalar operations with two source operands for CDNA
    architecture.
  }];
  let inputs = (ins SOP2SSrc:$src0, SOP2SSrc:$src1);
  let outputs = (ins OptionalReg<AnyRegOf<[SGPRType, SGPRRangeType]>>:$sdst);
  // TODO: add modifiers.
  let assemblyFormat = [{
    custom<Opcode>($opcode) (`outs` $sdst^)? `ins` $src0 `,` $src1 attr-dict
    `:` type(operands)
  }];
}

class SOP2Constraints<int sz, bit hasSdst> {
  dag constraints = (ins
    /*ins=*/
      RegImmOperand<SOP2SSrc, sz>:$src0,
      RegImmOperand<SOP2SSrc, sz>:$src1,
    /*outs=*/
      !if(hasSdst, RegImmOperand<AnyRegOf<[SGPRType, SGPRRangeType]>, sz>, NotPresent):$sdst
  );
}

class SOP2Inst<string _mnemonic, list<ISAVersion> _isa>
    : AMDInst<_mnemonic, _mnemonic, SOP2Op, _isa> {
  // Default constraints.
  let constraints = SOP2Constraints<1, 1>.constraints;
  // Default builders.
  let cppBuilder = DefaultCppBuilder<(ins
    CppValue:$sdst, CppValue:$src0, CppValue:$src1)>;
  let pythonBuilder = DefaultPyBuilder<(ins
    PyValue:$sdst, PyValue:$src0, PyValue:$src1)>;
  // Default ASM.
  let asmFormat = [SingleAsmVariant<"$sdst, $src0, $src1">];
}

//===----------------------------------------------------------------------===//
// SOP2Op instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Scalar add operations
//===----------------------------------------------------------------------===//

def SAddU32 : SOP2Inst<"s_add_u32", [CDNA3]> {
  let summary = "SOP2 add two unsigned 32-bit integers";
}

def SSubU32 : SOP2Inst<"s_sub_u32", [CDNA3]> {
  let summary = "SOP2 subtract unsigned 32-bit integers";
}

def SAddI32 : SOP2Inst<"s_add_i32", [CDNA3]> {
  let summary = "SOP2 add two signed 32-bit integers";
}

def SSubI32 : SOP2Inst<"s_sub_i32", [CDNA3]> {
  let summary = "SOP2 subtract signed 32-bit integers";
}

//===----------------------------------------------------------------------===//
// Scalar add/subtract with carry operations
//===----------------------------------------------------------------------===//

def SAddcU32 : SOP2Inst<"s_addc_u32", [CDNA3]> {
  let summary = "SOP2 add two unsigned 32-bit integers with carry-in";
}

def SSubbU32 : SOP2Inst<"s_subb_u32", [CDNA3]> {
  let summary = "SOP2 subtract unsigned 32-bit integers with borrow";
}

//===----------------------------------------------------------------------===//
// Scalar min/max operations
//===----------------------------------------------------------------------===//

def SMinI32 : SOP2Inst<"s_min_i32", [CDNA3]> {
  let summary = "SOP2 minimum of two signed 32-bit integers";
}

def SMinU32 : SOP2Inst<"s_min_u32", [CDNA3]> {
  let summary = "SOP2 minimum of two unsigned 32-bit integers";
}

def SMaxI32 : SOP2Inst<"s_max_i32", [CDNA3]> {
  let summary = "SOP2 maximum of two signed 32-bit integers";
}

def SMaxU32 : SOP2Inst<"s_max_u32", [CDNA3]> {
  let summary = "SOP2 maximum of two unsigned 32-bit integers";
}

//===----------------------------------------------------------------------===//
// Scalar conditional select operations
//===----------------------------------------------------------------------===//

def SCselectB32 : SOP2Inst<"s_cselect_b32", [CDNA3]> {
  let summary = "SOP2 conditional select 32-bit value based on SCC";
}

def SCselectB64 : SOP2Inst<"s_cselect_b64", [CDNA3]> {
  let summary = "SOP2 conditional select 64-bit value based on SCC";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

//===----------------------------------------------------------------------===//
// Scalar bitwise operations
//===----------------------------------------------------------------------===//

def SAndB32 : SOP2Inst<"s_and_b32", [CDNA3]> {
  let summary = "SOP2 bitwise AND of two 32-bit values";
}

def SAndB64 : SOP2Inst<"s_and_b64", [CDNA3]> {
  let summary = "SOP2 bitwise AND of two 64-bit values";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SOrB32 : SOP2Inst<"s_or_b32", [CDNA3]> {
  let summary = "SOP2 bitwise OR of two 32-bit values";
}

def SOrB64 : SOP2Inst<"s_or_b64", [CDNA3]> {
  let summary = "SOP2 bitwise OR of two 64-bit values";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SXorB32 : SOP2Inst<"s_xor_b32", [CDNA3]> {
  let summary = "SOP2 bitwise XOR of two 32-bit values";
}

def SXorB64 : SOP2Inst<"s_xor_b64", [CDNA3]> {
  let summary = "SOP2 bitwise XOR of two 64-bit values";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SAndn2B32 : SOP2Inst<"s_andn2_b32", [CDNA3]> {
  let summary = "SOP2 bitwise AND with negated second operand (32-bit)";
}

def SAndn2B64 : SOP2Inst<"s_andn2_b64", [CDNA3]> {
  let summary = "SOP2 bitwise AND with negated second operand (64-bit)";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SOrn2B32 : SOP2Inst<"s_orn2_b32", [CDNA3]> {
  let summary = "SOP2 bitwise OR with negated second operand (32-bit)";
}

def SOrn2B64 : SOP2Inst<"s_orn2_b64", [CDNA3]> {
  let summary = "SOP2 bitwise OR with negated second operand (64-bit)";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SNandB32 : SOP2Inst<"s_nand_b32", [CDNA3]> {
  let summary = "SOP2 bitwise NAND of two 32-bit values";
}

def SNandB64 : SOP2Inst<"s_nand_b64", [CDNA3]> {
  let summary = "SOP2 bitwise NAND of two 64-bit values";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SNorB32 : SOP2Inst<"s_nor_b32", [CDNA3]> {
  let summary = "SOP2 bitwise NOR of two 32-bit values";
}

def SNorB64 : SOP2Inst<"s_nor_b64", [CDNA3]> {
  let summary = "SOP2 bitwise NOR of two 64-bit values";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SXnorB32 : SOP2Inst<"s_xnor_b32", [CDNA3]> {
  let summary = "SOP2 bitwise XNOR of two 32-bit values";
}

def SXnorB64 : SOP2Inst<"s_xnor_b64", [CDNA3]> {
  let summary = "SOP2 bitwise XNOR of two 64-bit values";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

//===----------------------------------------------------------------------===//
// Scalar shift operations
//===----------------------------------------------------------------------===//

def SLshlB32 : SOP2Inst<"s_lshl_b32", [CDNA3]> {
  let summary = "SOP2 logical shift left 32-bit value";
}

def SLshlB64 : SOP2Inst<"s_lshl_b64", [CDNA3]> {
  let summary = "SOP2 logical shift left 64-bit value";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SLshrB32 : SOP2Inst<"s_lshr_b32", [CDNA3]> {
  let summary = "SOP2 logical shift right 32-bit value";
}

def SLshrB64 : SOP2Inst<"s_lshr_b64", [CDNA3]> {
  let summary = "SOP2 logical shift right 64-bit value";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SAshrI32 : SOP2Inst<"s_ashr_i32", [CDNA3]> {
  let summary = "SOP2 arithmetic shift right 32-bit value";
}

def SAshrI64 : SOP2Inst<"s_ashr_i64", [CDNA3]> {
  let summary = "SOP2 arithmetic shift right 64-bit value";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

//===----------------------------------------------------------------------===//
// Scalar bitfield operations
//===----------------------------------------------------------------------===//

def SBfmB32 : SOP2Inst<"s_bfm_b32", [CDNA3]> {
  let summary = "SOP2 create 32-bit bitfield mask";
}

def SBfmB64 : SOP2Inst<"s_bfm_b64", [CDNA3]> {
  let summary = "SOP2 create 64-bit bitfield mask";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SBfeU32 : SOP2Inst<"s_bfe_u32", [CDNA3]> {
  let summary = "SOP2 extract unsigned bitfield from 32-bit value";
}

def SBfeI32 : SOP2Inst<"s_bfe_i32", [CDNA3]> {
  let summary = "SOP2 extract signed bitfield from 32-bit value";
}

def SBfeU64 : SOP2Inst<"s_bfe_u64", [CDNA3]> {
  let summary = "SOP2 extract unsigned bitfield from 64-bit value";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

def SBfeI64 : SOP2Inst<"s_bfe_i64", [CDNA3]> {
  let summary = "SOP2 extract signed bitfield from 64-bit value";
  let constraints = SOP2Constraints<2, 1>.constraints;
}

//===----------------------------------------------------------------------===//
// Scalar multiply operations
//===----------------------------------------------------------------------===//

def SMulI32 : SOP2Inst<"s_mul_i32", [CDNA3]> {
  let summary = "SOP2 multiply two signed 32-bit integers";
}

def SMulHiU32 : SOP2Inst<"s_mul_hi_u32", [CDNA3]> {
  let summary = "SOP2 multiply unsigned 32-bit integers, return high 32 bits";
}

def SMulHiI32 : SOP2Inst<"s_mul_hi_i32", [CDNA3]> {
  let summary = "SOP2 multiply signed 32-bit integers, return high 32 bits";
}

//===----------------------------------------------------------------------===//
// Scalar special operations
//===----------------------------------------------------------------------===//

def SCbranchGFork : SOP2Inst<"s_cbranch_g_fork", [CDNA3]> {
  let summary = "SOP2 conditional branch using branch-stack";
  let constraints = (ins NotPresent:$sdst);
  let cppBuilder = OpBuilder<(ins CppValue:$src0, CppValue:$src1), [{
      return $_create($_builder, $_loc, $_opcode, nullptr, src0, src1);
    }]>;
  let pythonBuilder = OpBuilder<(ins PyValue:$src0, PyValue:$src1), [{
      return $_create(opcode=$_opcode, vdst0=None, src0=src0, src1=src1, $_lastArgs)
    }]>;
  let asmFormat = [SingleAsmVariant<"$src0, $src1">];
}

def SAbsdiffI32 : SOP2Inst<"s_absdiff_i32", [CDNA3]> {
  let summary = "SOP2 absolute difference of two signed 32-bit integers";
}

//===----------------------------------------------------------------------===//
// Scalar shift-add operations
//===----------------------------------------------------------------------===//

def SLshl1AddU32 : SOP2Inst<"s_lshl1_add_u32", [CDNA3]> {
  let summary = "SOP2 shift left by 1 and add unsigned 32-bit";
}

def SLshl2AddU32 : SOP2Inst<"s_lshl2_add_u32", [CDNA3]> {
  let summary = "SOP2 shift left by 2 and add unsigned 32-bit";
}

def SLshl3AddU32 : SOP2Inst<"s_lshl3_add_u32", [CDNA3]> {
  let summary = "SOP2 shift left by 3 and add unsigned 32-bit";
}

def SLshl4AddU32 : SOP2Inst<"s_lshl4_add_u32", [CDNA3]> {
  let summary = "SOP2 shift left by 4 and add unsigned 32-bit";
}

//===----------------------------------------------------------------------===//
// Scalar pack operations
//===----------------------------------------------------------------------===//

def SPackLlB32B16 : SOP2Inst<"s_pack_ll_b32_b16", [CDNA3]> {
  let summary = "SOP2 pack two low 16-bit values into 32-bit";
}

def SPackLhB32B16 : SOP2Inst<"s_pack_lh_b32_b16", [CDNA3]> {
  let summary = "SOP2 pack low and high 16-bit values into 32-bit";
}

def SPackHhB32B16 : SOP2Inst<"s_pack_hh_b32_b16", [CDNA3]> {
  let summary = "SOP2 pack two high 16-bit values into 32-bit";
}

#endif // AMDGCN_INST_SOP2_TD
