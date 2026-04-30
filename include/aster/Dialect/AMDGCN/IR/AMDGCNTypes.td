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
include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNRegisterTypeInterface.td"
include "aster/Interfaces/DependentOpInterface.td"

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class AMDGCN_RegisterDef<string name, string typeMnemonic, list<Trait> traits = []>
    : AMDGCN_Type<name, typeMnemonic, traits # [AMDGCNRegisterTypeInterface]>,
      PrintOperand {
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
    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::AGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return AGPRType::get(getContext(), range);
    }
    int64_t getSizeInBits() const {
      return static_cast<int64_t>(getRange().size()) * 32;
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
    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::SGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return SGPRType::get(getContext(), range);
    }
    int64_t getSizeInBits() const {
      return static_cast<int64_t>(getRange().size()) * 32;
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

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return RegisterKind::VGPR;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return VGPRType::get(getContext(), range);
    }
    int64_t getSizeInBits() const {
      return static_cast<int64_t>(getRange().size()) * 32;
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
class SREGBase<string name, string mnemonic, string kind>
    : AMDGCN_RegisterDef<name, mnemonic, [MemRefElementTypeInterface]> {
  let summary = kind # " special register type";
  let parameters = (ins
    DefaultValuedParameter<"Register", "Register()">:$reg
  );
  let assemblyFormat = "(`<` $reg^ `>`)?";
  let builders = [
    TypeBuilder<(ins "Register":$reg), [{
      return $_get($_ctxt, normalizeRegister(reg));
    }]>
  ];
  let skipDefaultBuilders = 1;
  string declarations = StrSubst<[{
    /// The register kind for this SREG type.
    static constexpr RegisterKind kRegisterKind = RegisterKind::$kind;
    }], [VarRepl<"kind", kind>]>.result;
  let extraClassDeclaration = declarations # [{
    static Register normalizeRegister(Register reg) {
      if (reg.getSemantics() == RegisterSemantics::Allocated)
        return Register(0);
      return reg;
    }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    RegisterRange getAsRange() const {
      return RegisterRange(getReg(), 1);
    }
    RegisterKind getRegisterKind() const {
      return kRegisterKind;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      assert(range.size() == 1 && "SREG type can only clone single register");
      return get(getContext(), range.begin());
    }
    int64_t getSizeInBits() const {
      switch (kRegisterKind) {
      case RegisterKind::VCC:
      case RegisterKind::EXEC:
        return 64;
      case RegisterKind::VCCZ:
      case RegisterKind::EXECZ:
      case RegisterKind::SCC:
        return 1;
      default:
        return 32;
      }
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

//===----------------------------------------------------------------------===//
// Misc types
//===----------------------------------------------------------------------===//

def LDSBufferType : AMDGCN_Type<"LDSBuffer", "lds_buffer",
    [MemRefElementTypeInterface]> {
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

#endif // ASTER_AMDGCN_TYPES_TD
