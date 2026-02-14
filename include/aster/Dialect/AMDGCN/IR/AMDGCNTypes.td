//===- AMDGCNTypes.td - AMDGCN Types ------------------------*- tablegen-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the amdgcn dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_TYPES_TD
#define ASTER_AMDGCN_TYPES_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNTypeInterfaces.td"
include "aster/Interfaces/DependentOpInterface.td"

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class AMDGCN_RegisterDef<string name, string typeMnemonic, list<Trait> traits = []>
    : AMDGCN_Type<name, typeMnemonic, traits # [AMDGCNRegisterTypeInterface]>,
      AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

//===----------------------------------------------------------------------===//
// AGPR like types
//===----------------------------------------------------------------------===//

def AGPRType : AMDGCN_RegisterDef<"AGPR", "agpr", [MemRefElementTypeInterface]> {
  let summary = "AGPR type";
  let parameters = (ins
    DefaultValuedParameter<"RegisterRange", "RegisterRange()">:$range
  );
  let assemblyFormat = "(`<` $range^ `>`)?";
  let builders = [
    TypeBuilder<(ins CArg<"Register", "Register()">:$reg, CArg<"int16_t", "1">:$size,
        CArg<"int16_t", "1">:$alignment), [{
      return $_get($_ctxt, RegisterRange(reg, size, alignment));
    }]>
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    Register getReg() const { return getRange().begin(); }

    /// Returns true if the register is relocatable.
    bool isRelocatable() const { return getReg().isRelocatable(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return getRange().size() > 1; }
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::AGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return AGPRType::get(getContext(), range);
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return AGPRType::get(getContext(), RegisterRange(reg, 1));
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

//===----------------------------------------------------------------------===//
// SGPR like types
//===----------------------------------------------------------------------===//

def SGPRType : AMDGCN_RegisterDef<"SGPR", "sgpr", [MemRefElementTypeInterface]> {
  let summary = "SGPR type";
  let parameters = (ins
    DefaultValuedParameter<"RegisterRange", "RegisterRange()">:$range
  );
  let assemblyFormat = "(`<` $range^ `>`)?";
  let builders = [
    TypeBuilder<(ins CArg<"Register", "Register()">:$reg, CArg<"int16_t", "1">:$size,
        CArg<"int16_t", "1">:$alignment), [{
      return $_get($_ctxt, RegisterRange(reg, size, alignment));
    }]>
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    Register getReg() const { return getRange().begin(); }

    /// Returns true if the register is relocatable.
    bool isRelocatable() const { return getReg().isRelocatable(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return getRange().size() > 1; }
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::SGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return SGPRType::get(getContext(), range);
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return SGPRType::get(getContext(), RegisterRange(reg, 1));
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

//===----------------------------------------------------------------------===//
// VGPR like types
//===----------------------------------------------------------------------===//

def VGPRType : AMDGCN_RegisterDef<"VGPR", "vgpr", [MemRefElementTypeInterface]> {
  let summary = "VGPR type";
  let parameters = (ins
    DefaultValuedParameter<"RegisterRange", "RegisterRange()">:$range
  );
  let assemblyFormat = "(`<` $range^ `>`)?";
  let builders = [
    TypeBuilder<(ins CArg<"Register", "Register()">:$reg, CArg<"int16_t", "1">:$size,
        CArg<"int16_t", "1">:$alignment), [{
      return $_get($_ctxt, RegisterRange(reg, size, alignment));
    }]>
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    Register getReg() const { return getRange().begin(); }

    /// Returns true if the register is relocatable.
    bool isRelocatable() const { return getRange().begin().isRelocatable(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return getRange().size() > 1; }
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::VGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return VGPRType::get(getContext(), range);
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return VGPRType::get(getContext(), RegisterRange(reg, 1));
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

//===----------------------------------------------------------------------===//
// SREG like types
//===----------------------------------------------------------------------===//

/// Special registers to model state.
def SREGType : AMDGCN_RegisterDef<"SREG", "sreg", [MemRefElementTypeInterface]> {
  let summary = "SREG type";
  let parameters = (ins
    DefaultValuedParameter<"Register", "Register()">:$reg,
    "SregKind":$kind);
  let assemblyFormat = "`<`$kind (`,` $reg^)?`>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// Returns true if the register is relocatable.
    bool isRelocatable() const { return getReg().isRelocatable(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return false; }
    RegisterRange getAsRange() const {
      return RegisterRange(getReg(), 1);
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::SREG;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      assert(range.size() == 1 && "SREG type can only clone single register");
      return SREGType::get(getContext(), range.begin(), getKind());
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return SREGType::get(getContext(), reg, getKind());
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

/// Special registers to model state.
class SREGBase<string name, string mnemonic, string kind>
    : AMDGCN_RegisterDef<name, mnemonic, [MemRefElementTypeInterface]> {
  let summary = kind # " special register type";
  let assemblyFormat = "";
  string declarations = StrSubst<[{
    /// The register kind for this SREG type.
    static constexpr RegisterKind kRegisterKind = RegisterKind::$kind;
    }], [VarRepl<"kind", kind>]>.result;
  let extraClassDeclaration = declarations # [{
    /// Returns true if the register is relocatable.
    bool isRelocatable() const { return false; }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return false; }
    RegisterRange getAsRange() const {
      return RegisterRange(Register(0), 1);
    }
    RegisterKind getRegisterKind() const {
      return kRegisterKind;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return get(getContext());
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return get(getContext());
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const {
      return SREGResource::get();
    }
  }];
}

// Define all special registers.
foreach sreg = SpecialRegisters in {
  def sreg#Type : SREGBase<sreg.typeName, sreg.typeMnemonic, sreg.name>;
}

class SReg<string name> : Type<CPred<"::mlir::isa<::mlir::aster::amdgcn::" # name # "Type>($_self)">,
    name # " special register type",
    "::mlir::aster::amdgcn::" # name # "Type"
  >, AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

//===----------------------------------------------------------------------===//
// Misc types
//===----------------------------------------------------------------------===//

def LDSBufferType : AMDGCN_Type<"LDSBuffer", "lds_buffer"> {
  let summary = "LDS buffer type";
  let description = [{
    A type representing a buffer allocated in the Local Data Store (LDS).
  }];
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Token types
//===----------------------------------------------------------------------===//

def ReadTokenType : AMDGCN_Type<"ReadToken", "read_token",
    [TokenDependencyTypeInterface, MemRefElementTypeInterface]> {
  let summary = "Read token type";
  let description = [{
    A token type representing a memory read operation. Parametrized by the
    memory instruction kind (flat, constant, or shared).
  }];
  let parameters = (ins "MemoryInstructionKind":$kind);
  let assemblyFormat = "`<` $kind `>`";
}

def WriteTokenType : AMDGCN_Type<"WriteToken", "write_token",
    [TokenDependencyTypeInterface, MemRefElementTypeInterface]> {
  let summary = "Write token type";
  let description = [{
    A token type representing a memory write operation. Parametrized by the
    memory instruction kind (flat, constant, or shared).
  }];
  let parameters = (ins "MemoryInstructionKind":$kind);
  let assemblyFormat = "`<` $kind `>`";
}

//===----------------------------------------------------------------------===//
// Common type constraints
//===----------------------------------------------------------------------===//

/// A type that matches any register type.
def AnyReg : Type<CPred<"true">, "any register type",
    "::mlir::aster::amdgcn::AMDGCNRegisterTypeInterface"
  >, AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

/// A type that matches any of the given register types.
class AnyRegOf<list<Type> types> : AnyTypeOf<types, "register type",
    "::mlir::aster::amdgcn::AMDGCNRegisterTypeInterface"
  >, AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

/// A sized SGPR range.
class SizedGPRRange<list<Type> types, list<int> size> :
  ConfinedType<AnyRegOf<types>, [
    CPred<"::mlir::aster::amdgcn::hasSize($_self, {" # !interleave(size, ", ") # "})">
  ], "sized GPR range">, AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

/// A type that matches an optional register type.
class OptionalReg<Type type> : Optional<type>, AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

/// A type that matches either a register or an immediate type.
class RegOrImm<list<Type> regTypes, list<Type> immTypes>
    : AnyTypeOf<regTypes # immTypes>, AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

/// A type that matches either a register or an immediate type.
class OptionalRegOrImm<list<Type> regTypes, list<Type> immTypes>
    : Optional<AnyTypeOf<regTypes # immTypes>>, AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

/// A 32-bit offset.
def OptionalOffsetType : Optional<I32>, AsmArgFormat {
  let asmParser = "$_parser.parseOffsetOperand($_self);";
  let asmPrinter = "$_printer.printOffsetOperand($_self);";
}

/// A label that can be branched to.
def BranchLabel : Successor<?, "">, AsmArgFormat {
  let asmParser = "$_parser.parseBranchLabel($_self);";
  let asmPrinter = "$_printer.printBranchLabel($_self);";
}

//===----------------------------------------------------------------------===//
// Register like types
//===----------------------------------------------------------------------===//

/// A register like type that matches AGPR, SGPR, VGPR, or SREG types.
def RegisterLike :  Type<CPred<"::mlir::aster::amdgcn::isRegisterLike($_self)">,
    "register like type", "::mlir::aster::amdgcn::AMDGCNRegisterTypeInterface"
  >, AsmArgFormat {
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
  >, AsmArgFormat {
  let asmParser = "$_parser.parseOperand($_self);";
  let asmPrinter = "$_printer.printOperand($_self);";
}

#endif // ASTER_AMDGCN_TYPES_TD
