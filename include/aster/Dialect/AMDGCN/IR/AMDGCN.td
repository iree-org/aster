//===-- AMDGCN.td - AMDGCN target definitions -----------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines AMDGCN target definitions.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_TD
#define ASTER_AMDGCN_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnumAttrs.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNModifiers.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/Constraints.td"

//===----------------------------------------------------------------------===//
// Inst helpers
//===----------------------------------------------------------------------===//

/// Defines a default C++ builder for an instruction op.
class DefaultCppBuilder<dag params> : OpBuilder<params, !if(
!gt(!size(params), 0), [{
  return $_create($_builder, $_loc, $_opcode, $_args);
}], [{
  return $_create($_builder, $_loc, $_opcode);
}])>;

/// Defines a default python builder for an instruction op.
class DefaultPyBuilder<dag params> : OpBuilder<params, !if(
!gt(!size(params), 0), [{
  return $_create(opcode=$_opcode, $_args, $_lastArgs)
}], [{
  return $_create(opcode=$_opcode, $_lastArgs)
}])>;

/// Shorthand for C++ values in the instruction builders.
defvar CppValue = "::mlir::Value";

/// Shorthand for Python values in the instruction builders.
defvar PyValue = "$_ir.Value";
defvar PyBlock = "$_ir.Block";

/// Helper for C++/python integer arguments on builders.
class IntArg<int v> : CArg<"int", !cast<string>(v)>;

class IntOrValueArg<int v> : CArg<"$_typing.Union[int, $_ir.Value]", !cast<string>(v)>;

/// Helper for python boolean arguments on builders.
class PyBool<bit v> : CArg<"bool", !if(v, "True", "False")>;

/// A helper for when an instruction has a single assembly variant.
class SingleAsmVariant<string format> : AsmVariant<TruePred, format>;

//===----------------------------------------------------------------------===//
// Inst operands helpers
//===----------------------------------------------------------------------===//

/// Type with description.
/// This is hack to use use the description field of Type to provide custom error
/// messages for instruction operand constraints.
class TypeWithDescription<Pred condition, string summary, string _desc = "",
                        string cppType = "::mlir::Type">
  : Type<condition, summary, cppType> {
  let description = _desc;
}

/// Register operand constraint. It check the number of registers and alignment.
class RegisterOperand<
    list<Type> validRegTypes, int numRegs, int alignment = -1, bit optional = 0
  > : InstConstraint<> {
  code cPred = !subst("$alignment", !cast<string>(alignment),
                      !subst("$numRegs", !cast<string>(numRegs),
                      "isValidRegisterType($_self, $numRegs, $alignment)"));
  code desc = !subst("$alignment", !cast<string>(alignment),
                      !subst("$numRegs", !cast<string>(numRegs),
                      "$_argName operand must have $numRegs registers for $_mnemonic, got "));
  let constraints = [
    !if(optional, Optional<AnyTypeOf<validRegTypes>>, AnyTypeOf<validRegTypes>),
    !if(optional,
      Optional<TypeWithDescription<CPred<cPred>, desc, "getRangeSize($_self)", "AMDGCNRegisterTypeInterface">>,
      TypeWithDescription<CPred<cPred>, desc, "getRangeSize($_self)", "AMDGCNRegisterTypeInterface">)
  ];
}

/// Register or Imm operand constraint. It check the number of registers and alignment.
class RegImmOperand<Type validType, int numRegs, bit optional = 0> :
    InstConstraint<> {
  code cPred = !subst("$numRegs", !cast<string>(numRegs),
                      "isValidRegImmType($_self, $numRegs)");
  code desc = !subst("$numRegs", !cast<string>(numRegs),
                     "$_argName operand must have $numRegs or be an Imm registers for $_mnemonic, got ");
  let constraints = [
    !if(optional, Optional<validType>, validType),
    !if(optional,
      Optional<TypeWithDescription<CPred<cPred>, desc, "getRangeSize($_self)", "Type">>,
      TypeWithDescription<CPred<cPred>, desc, "getRangeSize($_self)", "Type">)
  ];
}

/// SGPR operand constraint.
class SGPROperand<int numRegs, bit optional = 0, int alignment = -1> :
    RegisterOperand<[SGPRType, SGPRRangeType], numRegs, alignment, optional> {
}

/// VGPR operand constraint.
class VGPROperand<int numRegs, bit optional = 0, int alignment = -1> :
    RegisterOperand<[VGPRType, VGPRRangeType], numRegs, alignment, optional> {
}

/// ACC operand constraint.
class AccOperand<int numRegs, bit optional = 0, int alignment = -1> :
    RegisterOperand<[VGPRType, AGPRType, VGPRRangeType, AGPRRangeType], numRegs, alignment, optional> {
}

/// Addr operand constraint.
class AddrOperand<int numRegs, bit optional = 0, int alignment = -1> :
    RegisterOperand<[SGPRType, VGPRType, SGPRRangeType, VGPRRangeType], numRegs, alignment, optional> {
}

/// The optional value must not be present.
def NotPresent : InstConstraint<[
  TypeConstraint<
    CPred<"$_self == nullptr">, "expected `$_argName` to not be present">
]>;

/// The optional value must be present.
def Present : InstConstraint<[
  TypeConstraint<
    CPred<"$_self != nullptr">, "expected `$_argName` to be present">
]>;

/// A register or immediate operand that can be either SGPR or VGPR type
def VIntSrc : RegOrImm<[SGPRType, VGPRType], [I32]>;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

def InvalidCase : EnumCaseInfo<"invalid", "Invalid">;

//===----------------------------------------------------------------------===//
// ISA Versions
//===----------------------------------------------------------------------===//

def CDNA3 : ISAVersion<"cdna3"> { let description = "CDNA 3 ISA version"; }

def CDNA4 : ISAVersion<"cdna4"> { let description = "CDNA 4 ISA version"; }

def RDNA4 : ISAVersion<"rdna4"> { let description = "RDNA 4 ISA version"; }

def IsaVersions
    : AutoI32EnumCases<"ISAVersion",
                       "AMDGCN ISA Versions", [InvalidCase, CDNA3, CDNA4, RDNA4]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def IsaVersionAttr : AMDGCN_EnumAttr<"IsaVersion", "isa", IsaVersions>;

//===----------------------------------------------------------------------===//
// Target definitions
//===----------------------------------------------------------------------===//

def GFX940 : Target<"gfx940", CDNA3> { let description = "GFX940 target"; }

def GFX942 : Target<"gfx942", CDNA3> { let description = "GFX942 target"; }

def GFX950 : Target<"gfx950", CDNA4> { let description = "GFX950 target"; }

def GFX1201 : Target<"gfx1201", RDNA4> { let description = "GFX1201 target"; }

def AMDGCNTargets
    : AutoI32EnumCases<
          "Target", "AMDGCN Targets", [InvalidCase, GFX940, GFX942, GFX950, GFX1201]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def AMDGCNTargetAttr : AMDGCN_EnumAttr<"Target", "target", AMDGCNTargets>;

#endif // ASTER_AMDGCN_TD
