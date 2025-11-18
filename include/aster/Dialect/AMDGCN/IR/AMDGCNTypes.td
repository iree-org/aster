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
// Generic register type
//===----------------------------------------------------------------------===//

def GenericRegisterType
    : AMDGCN_RegisterDef<"GGPR", "ggpr", [MemRefElementTypeInterface]> {
  let summary = "Generic register type";
  let parameters = (ins
    DefaultValuedParameter<"RegisterRange", "RegisterRange()">:$range,
    OptionalParameter<"std::optional<bool>">:$is_uniform
  );
  let assemblyFormat = "`<` $range (`,` `uniform` `=` $is_uniform^)? `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// Returns true if the register range is relocatable.
    bool isRelocatable() const { return getRange().begin().isRelocatable(); }
    /// Returns the size of the register range.
    int64_t size() const { return getRange().size(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return true; }
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::GGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return GGPRType::get(getContext(), range, getIsUniform());
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return GGPRType::get(getContext(), RegisterRange(reg, 1), getIsUniform());
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

//===----------------------------------------------------------------------===//
// AGPR like types
//===----------------------------------------------------------------------===//

def AGPRType : AMDGCN_RegisterDef<"AGPR", "agpr", [MemRefElementTypeInterface]> {
  let summary = "AGPR type";
  let parameters = (ins DefaultValuedParameter<"Register", "Register()">:$reg);
  let assemblyFormat = "(`<` $reg^ `>`)?";
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
      return RegisterKind::AGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      assert(range.size() == 1 && "AGPR type can only clone single register");
      return AGPRType::get(getContext(), range.begin());
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return AGPRType::get(getContext(), reg);
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

def AGPRRangeType
    : AMDGCN_RegisterDef<"AGPRRange", "agpr_range", [MemRefElementTypeInterface]> {
  let summary = "AGPR range type";
  let parameters = (ins DefaultValuedParameter<"RegisterRange", "RegisterRange()">:$range);
  let assemblyFormat = "(`<` $range^ `>`)?";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// Returns true if the register range is relocatable.
    bool isRelocatable() const { return getRange().begin().isRelocatable(); }
    /// Returns the size of the register range.
    int64_t size() const { return getRange().size(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return true; }
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::AGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return AGPRRangeType::get(getContext(), range);
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return AGPRType::get(getContext(), reg);
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
  let parameters = (ins DefaultValuedParameter<"Register", "Register()">:$reg);
  let assemblyFormat = "(`<` $reg^ `>`)?";
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
      return RegisterKind::SGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      assert(range.size() == 1 && "SGPR type can only clone single register");
      return SGPRType::get(getContext(), range.begin());
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return SGPRType::get(getContext(), reg);
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

def SGPRRangeType
    : AMDGCN_RegisterDef<"SGPRRange", "sgpr_range", [MemRefElementTypeInterface]> {
  let summary = "SGPR range type";
  let parameters = (ins DefaultValuedParameter<"RegisterRange", "RegisterRange()">:$range);
  let assemblyFormat = "(`<` $range^ `>`)?";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// Returns true if the register range is relocatable.
    bool isRelocatable() const { return getRange().begin().isRelocatable(); }
    /// Returns the size of the register range.
    int64_t size() const { return getRange().size(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return true; }
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::SGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return SGPRRangeType::get(getContext(), range);
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return SGPRType::get(getContext(), reg);
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
  let parameters = (ins DefaultValuedParameter<"Register", "Register()">:$reg);
  let assemblyFormat = "(`<` $reg^ `>`)?";
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
      return RegisterKind::VGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      assert(range.size() == 1 && "VGPR type can only clone single register");
      return VGPRType::get(getContext(), range.begin());
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return VGPRType::get(getContext(), reg);
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

def VGPRRangeType
    : AMDGCN_RegisterDef<"VGPRRange", "vgpr_range", [MemRefElementTypeInterface]> {
  let summary = "VGPR range type";
  let parameters = (ins DefaultValuedParameter<"RegisterRange", "RegisterRange()">:$range);
  let assemblyFormat = "(`<` $range^ `>`)?";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// Returns true if the register range is relocatable.
    bool isRelocatable() const { return getRange().begin().isRelocatable(); }
    /// Returns the size of the register range.
    int64_t size() const { return getRange().size(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    bool isRegisterRange() const { return true; }
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::VGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return VGPRRangeType::get(getContext(), range);
    }
    RegisterTypeInterface cloneRegisterType(Register reg) const {
      return VGPRType::get(getContext(), reg);
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
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
def OffsetType : I<32>, AsmArgFormat {
  let asmParser = "$_parser.parseOffsetOperand($_self);";
  let asmPrinter = "$_printer.printOffsetOperand($_self);";
}

//===----------------------------------------------------------------------===//
// Register like types
//===----------------------------------------------------------------------===//

/// A register like type that matches AGPR, SGPR, or VGPR types.
def RegisterLike : AnyRegOf<[AGPRType, SGPRType, VGPRType]> {
  let summary = "Register like types";
  let description = [{
    The register like types represent the different kinds of registers
    (AGPR, SGPR, VGPR) in the AMDGCN architecture.
  }];
}

/// A register like type that matches AGPRRange, SGPRRange, or VGPRRange types.
def RegisterRangeLike : AnyRegOf<[AGPRRangeType, SGPRRangeType, VGPRRangeType]> {
  let summary = "Register range like types";
  let description = [{
    The register range like types represent the different kinds of registers
    ranges (AGPR, SGPR, VGPR) in the AMDGCN architecture.
  }];
}

/// A register type that matches either a VGPR or an AGPR type.
def VGPROrAGPRType : AnyRegOf<[AGPRRangeType, VGPRRangeType]>;

#endif // ASTER_AMDGCN_TYPES_TD
