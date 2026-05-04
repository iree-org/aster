//===- AMDGCNTypeConstraints.td - AMDGCN Type Constraints ---*- tablegen-*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type constraints for the AMDGCN dialect. These are
// used by ops and instructions to constrain operand and result types.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_TYPE_CONSTRAINTS_TD
#define ASTER_AMDGCN_TYPE_CONSTRAINTS_TD

include "aster/IR/Inst.td"
include "aster/IR/Util.td"
include "aster/Interfaces/RegisterType.td"
include "aster/Interfaces/DependentOpInterface.td"

//===----------------------------------------------------------------------===//
// Register type constraints (AMDReg)
//===----------------------------------------------------------------------===//

class AMDReg<string cppType, string name> :
  Type<CPred<"::llvm::isa<::mlir::aster::amdgcn::" # cppType # ">($_self)">,
       name # " type", "::mlir::aster::amdgcn::" # cppType>,
  PrintOperand;

def AGPRType : AMDReg<"AGPRType", "AGPR">;
def VGPRType : AMDReg<"VGPRType", "VGPR">;
def SGPRType : AMDReg<"SGPRType", "SGPR">;

def VCCType : AMDReg<"VCCType", "VCC">;
def VCCLoType : AMDReg<"VCCLoType", "VCC Lo">;
def VCCHiType : AMDReg<"VCCHiType", "VCC Hi">;
def M0Type : AMDReg<"M0Type", "M0">;
def EXECType : AMDReg<"EXECType", "EXEC">;
def EXECLoType : AMDReg<"EXECLoType", "EXEC Lo">;
def EXECHiType : AMDReg<"EXECHiType", "EXEC Hi">;
def VCCZType : AMDReg<"VCCZType", "VCCZ">;
def EXECZType : AMDReg<"EXECZType", "EXECZ">;
def SCCType : AMDReg<"SCCType", "SCC">;

//===----------------------------------------------------------------------===//
// Special register constraint
//===----------------------------------------------------------------------===//

class SReg<string name> : Type<CPred<"::mlir::isa<::mlir::aster::amdgcn::" # name # "Type>($_self)">,
    name # " special register type",
    "::mlir::aster::amdgcn::" # name # "Type"
  >, PrintOperand {
}

//===----------------------------------------------------------------------===//
// Misc type constraints (for types only available as type definitions)
//===----------------------------------------------------------------------===//

def LDSBufferType : CPPType<"::mlir::aster::amdgcn::LDSBufferType",
    "LDS buffer type">,
  BuildableType<"$_builder.getType<::mlir::aster::amdgcn::LDSBufferType>()">;

def ReadTokenType : CPPType<"::mlir::aster::amdgcn::ReadTokenType",
    "read token type">;

def WriteTokenType : CPPType<"::mlir::aster::amdgcn::WriteTokenType",
    "write token type">;

def DSReadToken :
  CPPType<"::mlir::aster::amdgcn::ReadTokenType", "read token type">,
  InferrableType<"$_builder.getType<ReadTokenType>(MemoryInstructionKind::Shared)">;

def DSWriteToken :
  CPPType<"::mlir::aster::amdgcn::WriteTokenType", "write token type">,
  InferrableType<"$_builder.getType<WriteTokenType>(MemoryInstructionKind::Shared)">;

def ConstantReadToken :
  CPPType<"::mlir::aster::amdgcn::ReadTokenType", "read token type">,
  InferrableType<"$_builder.getType<ReadTokenType>(MemoryInstructionKind::Constant)">;

def ConstantWriteToken :
  CPPType<"::mlir::aster::amdgcn::WriteTokenType", "write token type">,
  InferrableType<"$_builder.getType<WriteTokenType>(MemoryInstructionKind::Constant)">;

def FlatReadToken :
  CPPType<"::mlir::aster::amdgcn::ReadTokenType", "read token type">,
  InferrableType<"$_builder.getType<ReadTokenType>(MemoryInstructionKind::Flat)">;

def FlatWriteToken :
  CPPType<"::mlir::aster::amdgcn::WriteTokenType", "write token type">,
  InferrableType<"$_builder.getType<WriteTokenType>(MemoryInstructionKind::Flat)">;

//===----------------------------------------------------------------------===//
// Common type constraint classes
//===----------------------------------------------------------------------===//

/// A type that matches any register type.
def AnyReg : Type<CPred<"true">, "any register type",
    "::mlir::aster::amdgcn::AMDGCNRegisterTypeInterface"
  >, PrintOperand {
}

/// A type that matches any of the given register types.
class AnyRegOf<list<Type> types> :
  AnyTypeOf<types, "register type of: {" # !interleave(!foreach(type, types, type.summary), ", ") # "}",
    "::mlir::aster::amdgcn::AMDGCNRegisterTypeInterface">,
  PrintOperand;

/// A sized GPR range of a specific type.
class SizedGPRRangeOf<Type type, list<int> size> :
  ConfinedType<type, [
    CPred<"::mlir::aster::amdgcn::hasSize($_self, {" # !interleave(size, ", ") # "})">
  ], "sized GPR range of " # type.summary # " with sizes {" # !interleave(size, ", ") # "}">, PrintOperand {
}

/// Sized GPR range shortcuts.
class AGPROf<int numRegs> : SizedGPRRangeOf<AGPRType, [numRegs]>;
class SGPROf<int numRegs> : SizedGPRRangeOf<SGPRType, [numRegs]>;
class VGPROf<int numRegs> : SizedGPRRangeOf<VGPRType, [numRegs]>;

//===----------------------------------------------------------------------===//
// Optional and composite register constraints
//===----------------------------------------------------------------------===//

/// A type that matches an optional register type.
class OptionalReg<Type type> : Optional<type>, PrintOperand {
}

/// An optional operand.
class OptionalOperand<Type type> : Optional<type>, PrintOperand;

/// A type that matches either a register or an immediate type.
class RegOrImm<list<Type> regTypes, list<Type> immTypes>
    : AnyTypeOf<regTypes # immTypes>, PrintOperand {
}

/// A type that matches either a register or an immediate type (optional).
class OptionalRegOrImm<list<Type> regTypes, list<Type> immTypes>
    : Optional<AnyTypeOf<regTypes # immTypes>>, PrintOperand {
}

//===----------------------------------------------------------------------===//
// Register like types
//===----------------------------------------------------------------------===//

/// A register like type that matches AGPR, SGPR, VGPR, or SREG types.
def RegisterLike :  Type<CPred<"::mlir::aster::amdgcn::isRegisterLike($_self)">,
    "register like type", "::mlir::aster::amdgcn::AMDGCNRegisterTypeInterface"
  >, PrintOperand {
  let summary = "Register like types";
  let description = [{
    The register like types represent the different kinds of registers
    (AGPR, SGPR, VGPR, SREG) in the AMDGCN architecture.
  }];
}

/// A register like type that matches AGPRRange, SGPRRange, or VGPRRange types.
def RegisterRangeLike : AnyRegOf<[AGPRType, SGPRType, VGPRType]> {
  let summary = "Register range like types";
  let description = [{
    The register range like types represent the different kinds of registers
    ranges (AGPR, SGPR, VGPR) in the AMDGCN architecture.
  }];
}

/// A register type that matches either a VGPR or an AGPR type.
def VGPROrAGPRType : AnyRegOf<[AGPRType, VGPRType]>;

/// SCC register.
def SCCReg : Type<CPred<"cast<SREGType>($_self).getKind() == SregKind::Scc">,
    "SCC register",
    "::mlir::aster::amdgcn::SREGType"
  >, PrintOperand {
}

//===----------------------------------------------------------------------===//
// Offset and branch label constraints
//===----------------------------------------------------------------------===//

/// A 32-bit offset.
def OptionalOffsetType : Optional<I32>, ASMArgFormat {
  let asmPrinter = "$_printer.printOffsetOperand($_self);";
}

/// A label that can be branched to.
def BranchLabel : Successor<?, "">, ASMArgFormat {
  let asmPrinter = "$_printer.printBranchLabel($_self);";
}

//===----------------------------------------------------------------------===//
// Inline literal constraints
//===----------------------------------------------------------------------===//

class InlineLitFP<list<Type> types> : ConfinedType<
  AnyTypeOf<types, "an inline literal floating-point operand", "mlir::FloatType">, [
    CPred<[{
    true /*::mlir::aster::amdgcn::checkFloatConst($_selfValue, {-4.0, -2.0, -1.0, -0.5, 0.15915494, 0.5, 1.0, 2.0, 4.0})*/
  }]>
]>, PrintOperand;

class InlineLitInt<list<Type> types> : ConfinedType<
  AnyTypeOf<types, "an inline literal integer operand", "mlir::IntegerType">, [
    CPred<StrSubst<[{
      true /*::mlir::aster::amdgcn::checkIntConst($_selfValue, $_values)*/
    }], [VarRepl<"_values", "{" # !interleave([-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64], ", ") # "}">]>.result>
]>, PrintOperand;

class InlineLitIntPos<list<Type> types> : ConfinedType<AnyTypeOf<types, "an inline literal positive integer operand", "mlir::IntegerType">, [
    CPred<StrSubst<[{
      true /*::mlir::aster::amdgcn::checkIntConst($_selfValue, $_values)*/
    }], [VarRepl<"_values", "{" # !interleave([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64], ", ") # "}">]>.result>
]>, PrintOperand;

//===----------------------------------------------------------------------===//
// Integer type constraints
//===----------------------------------------------------------------------===//

class IntType<int bitWidth> : I<bitWidth>, ASMArgFormat, PrintOperand;

class OffsetType<int bitWidth, bit _isSigned = false> : ConfinedType<I<32>, [
    CPred<StrSubst<[{
      true /*::mlir::aster::amdgcn::checkOffsetConst($_selfValue, $_offsetWidth, $_isSigned)*/
    }], [IntRepl<"_offsetWidth", bitWidth>, IntRepl<"_isSigned", _isSigned>]>.result>
  ]>, PrintOperand {
  int offsetWidth = bitWidth;
  bit isSigned = _isSigned;
  let asmPrinter = "$_printer.printOffsetOperand($_self);";
}

class OptionalOffsetType<int bitWidth, bit _isSigned = false> : Optional<ConfinedType<I<32>, [
    CPred<StrSubst<[{
      true /*::mlir::aster::amdgcn::checkOffsetConst($_selfValue, $_offsetWidth, $_isSigned)*/
    }], [IntRepl<"_offsetWidth", bitWidth>, IntRepl<"_isSigned", _isSigned>]>.result>
  ]>>, PrintOperand {
  int offsetWidth = bitWidth;
  bit isSigned = _isSigned;
  let asmPrinter = "$_printer.printOffsetOperand($_self);";
}

//===----------------------------------------------------------------------===//
// Composite operand constraints
//===----------------------------------------------------------------------===//

class OneGPROrSRegOrConstOf<Type litType, Type dataType> : AnyOperandOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, litType, dataType]>;
class OneGPROrSRegOrLitOf<Type litType> : AnyOperandOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, litType]>;
class OneGPROrVCCM0OrFPConstOf<Type floatLit, Type fpConst> : AnyOperandOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, VCCZType, EXECZType, SCCType, floatLit, fpConst]>;
class OneGPROrVCCM0OrFPLitOf<Type floatLit> : AnyOperandOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, VCCZType, EXECZType, SCCType, floatLit]>;
class OneSGPROrSRegOrIntConstOf<Type intLit, Type intConst> : AnyOperandOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, intLit, intConst]>;
class OneSGPROrSRegOrLitConstOf<Type floatLit, Type intLit, Type fpConst, Type intConst> : AnyOperandOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, floatLit, intLit, fpConst, intConst]>;
class TwoGPROrSRegOrConstOf<Type litType, Type intConst> : AnyOperandOf<[VGPROf<2>, SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, litType, intConst]>;
class TwoGPROrVCCOrFPLitOf<Type floatLit> : AnyOperandOf<[VGPROf<2>, SGPROf<2>, VCCType, VCCZType, EXECZType, SCCType, floatLit]>;
class VGPROrAGPROf<int numRegs> : AnyRegOf<[VGPROf<numRegs>, AGPROf<numRegs>]>;
class VGPROrAGPROrLitOf<int numRegs, Type litType> : AnyOperandOf<[VGPROf<numRegs>, AGPROf<numRegs>, litType]>;

//===----------------------------------------------------------------------===//
// Concrete composite constraint defs
//===----------------------------------------------------------------------===//

def OneGPROrSRegOrConstType : AnyOperandOf<[VGPROf<1>, SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType, VCCZType, EXECZType, SCCType, IntType<32>, InlineLitInt<[IntType<32>]>, InlineLitFP<[F32]>, F32]>;
def OneSGPROrSRegType : AnyRegOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type, EXECLoType, EXECHiType]>;
def OneSGPROrVCCM0OrIntLitType : AnyOperandOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type, InlineLitIntPos<[IntType<32>]>]>;
def OneSGPROrVCCM0Type : AnyRegOf<[SGPROf<1>, VCCLoType, VCCHiType, M0Type]>;
def OneSGPROrVCCType : AnyRegOf<[SGPROf<1>, VCCLoType, VCCHiType]>;
def TwoGPROrSRegOrIntLitType : AnyOperandOf<[VGPROf<2>, SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, InlineLitInt<[IntType<32>]>]>;
def TwoGPROrVCCOrFPConstType : AnyOperandOf<[VGPROf<2>, SGPROf<2>, VCCType, VCCZType, EXECZType, SCCType, F64, InlineLitFP<[F64]>]>;
def TwoSGPROrSRegOrConstType : AnyOperandOf<[SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, IntType<64>, InlineLitInt<[IntType<64>]>, InlineLitFP<[F64]>, F64]>;
def TwoSGPROrSRegOrIntConstType : AnyOperandOf<[SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, IntType<64>, InlineLitInt<[IntType<64>]>]>;
def TwoSGPROrSRegOrLitType : AnyOperandOf<[SGPROf<2>, VCCType, EXECType, VCCZType, EXECZType, SCCType, InlineLitInt<[IntType<64>]>, InlineLitFP<[F64]>]>;
def TwoSGPROrSRegType : AnyRegOf<[SGPROf<2>, VCCType, EXECType]>;
def TwoSGPROrVCCType : AnyRegOf<[SGPROf<2>, VCCType]>;

#endif // ASTER_AMDGCN_TYPE_CONSTRAINTS_TD
