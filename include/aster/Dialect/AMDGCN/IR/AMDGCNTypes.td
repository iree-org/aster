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
// GP register types
//===----------------------------------------------------------------------===//

def GPRegTrait : NativeTypeTrait<"GPRegTrait"> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let extraConcreteClassDeclaration = [{
    Register getReg() const { return getRange().begin(); }

    //===------------------------------------------------------------------===//
    // RegisterTypeInterface
    //===------------------------------------------------------------------===//
    RegisterRange getAsRange() const {
      return getRange();
    }
    RegisterKind getRegisterKind() const {
      return kRegisterKind;
    }
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      return get(getContext(), range);
    }
    int64_t getSizeInBits() const {
      return static_cast<int64_t>(getRange().size()) * 32;
    }
    RegisterProps getProps() const {
      RegisterProps props = RegisterProps::IsGeneralPurpose |
                            RegisterProps::AcceptsValueSemantics |
                            RegisterProps::AcceptsUnallocatedSemantics |
                            RegisterProps::AcceptsAllocatedSemantics;
      if (getRange().size() > 1)
        return props | RegisterProps::IsComposite;
      return props | RegisterProps::AcceptsComposite;
    }
    RegisterTypeInterface getCompositeType(TypeRange types,
        std::optional<int16_t> alignment,
        llvm::function_ref<InFlightDiagnostic()> emitError) const;
    LogicalResult getSplitType(
        SmallVectorImpl<RegisterTypeInterface> &regs,
        llvm::function_ref<InFlightDiagnostic()> emitError) const;

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const;
  }];
}

class GPRegTypeBase<string name, string mnemonic>
    : AMDGCN_RegisterDef<name, mnemonic, [MemRefElementTypeInterface, GPRegTrait]> {
  let parameters = (ins
    DefaultValuedParameter<"RegisterRange", "RegisterRange()">:$range
  );
  let assemblyFormat = "(`<` $range^ `>`)?";
  let builders = [
    TypeBuilder<(ins CArg<"Register", "Register()">:$reg,
        CArg<"int16_t", "1">:$size,
        CArg<"int16_t", "1">:$alignment), [{
      return $_get($_ctxt, RegisterRange(reg, size, alignment));
    }]>
  ];
  let genVerifyDecl = 1;
}

def AGPRType : GPRegTypeBase<"AGPR", "agpr"> {
  let summary = "AGPR type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::AGPR;
  }];
}

def SGPRType : GPRegTypeBase<"SGPR", "sgpr"> {
  let summary = "SGPR type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::SGPR;
  }];
}

def VGPRType : GPRegTypeBase<"VGPR", "vgpr"> {
  let summary = "VGPR type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::VGPR;
  }];
}

//===----------------------------------------------------------------------===//
// SREG like types
//===----------------------------------------------------------------------===//

def SpecialRegTrait : NativeTypeTrait<"SpecialRegTrait"> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let extraConcreteClassDeclaration = [{
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
    RegisterTypeInterface cloneRegisterType(RegisterRange range) const {
      assert(range.size() == 1 && "SREG type can only clone single register");
      return get(getContext(), range.begin());
    }
    int64_t getSizeInBits() const {
      return kSizeInBits;
    }
    RegisterProps getProps() const {
      return kRegisterProps;
    }
    RegisterTypeInterface getCompositeType(TypeRange types,
        std::optional<int16_t> alignment,
        llvm::function_ref<InFlightDiagnostic()> emitError) const;
    LogicalResult getSplitType(
        SmallVectorImpl<RegisterTypeInterface> &regs,
        llvm::function_ref<InFlightDiagnostic()> emitError) const;

    //===------------------------------------------------------------------===//
    // AMDGCNRegisterTypeInterface
    //===------------------------------------------------------------------===//
    RegisterKind getRegisterKind() const {
      return kRegisterKind;
    }

    //===------------------------------------------------------------------===//
    // ResourceTypeInterface
    //===------------------------------------------------------------------===//
    Resource *getResource() const {
      return SREGResource::get();
    }
  }];
}

class SRegTypeBase<string name, string mnemonic>
    : AMDGCN_RegisterDef<name, mnemonic,
        [MemRefElementTypeInterface, SpecialRegTrait]> {
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
  let genVerifyDecl = 1;
}

/// Special registers to model state.
def VCCType : SRegTypeBase<"VCC", "vcc"> {
  let summary = "VCC special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::VCC;
    static constexpr int64_t kSizeInBits = 64;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::IsComposite |
        RegisterProps::AcceptsValueSemantics |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

def VCCLoType : SRegTypeBase<"VCCLo", "vcc_lo"> {
  let summary = "VCC_LO special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::VCC_LO;
    static constexpr int64_t kSizeInBits = 32;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::AcceptsComposite |
        RegisterProps::AcceptsValueSemantics |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

def VCCHiType : SRegTypeBase<"VCCHi", "vcc_hi"> {
  let summary = "VCC_HI special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::VCC_HI;
    static constexpr int64_t kSizeInBits = 32;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::AcceptsValueSemantics |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

/// VCCZ is a read-only status register. When used with value semantics, it
/// represents a frozen-in-time snapshot of the VCC zero flag at the point the
/// value was created. Value-semantic VCCZ may be promoted to SGPRs.
def VCCZType : SRegTypeBase<"VCCZ", "vccz"> {
  let summary = "VCCZ special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::VCCZ;
    static constexpr int64_t kSizeInBits = 1;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::IsStatusReg |
        RegisterProps::IsReadOnly |
        RegisterProps::AcceptsValueSemantics |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

def EXECType : SRegTypeBase<"EXEC", "exec"> {
  let summary = "EXEC special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::EXEC;
    static constexpr int64_t kSizeInBits = 64;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::IsComposite |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

def EXECLoType : SRegTypeBase<"EXECLo", "exec_lo"> {
  let summary = "EXEC_LO special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::EXEC_LO;
    static constexpr int64_t kSizeInBits = 32;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::AcceptsComposite |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

def EXECHiType : SRegTypeBase<"EXECHi", "exec_hi"> {
  let summary = "EXEC_HI special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::EXEC_HI;
    static constexpr int64_t kSizeInBits = 32;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

/// EXECZ is a read-only status register. When used with value semantics, it
/// represents a frozen-in-time snapshot of the EXEC zero flag at the point the
/// value was created. Value-semantic EXECZ may be promoted to SGPRs.
def EXECZType : SRegTypeBase<"EXECZ", "execz"> {
  let summary = "EXECZ special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::EXECZ;
    static constexpr int64_t kSizeInBits = 1;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::IsStatusReg |
        RegisterProps::IsReadOnly |
        RegisterProps::AcceptsValueSemantics |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

def M0Type : SRegTypeBase<"M0", "m0"> {
  let summary = "M0 special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::M0;
    static constexpr int64_t kSizeInBits = 32;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::AcceptsValueSemantics |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

def SCCType : SRegTypeBase<"SCC", "scc"> {
  let summary = "SCC special register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::SCC;
    static constexpr int64_t kSizeInBits = 1;
    static constexpr RegisterProps kRegisterProps =
        RegisterProps::IsStatusReg |
        RegisterProps::AcceptsValueSemantics |
        RegisterProps::AcceptsAllocatedSemantics;
  }];
}

// TTMP (trap-temporary) registers, special non-allocatable registers.
// Supports cluster_id ops on gfx1250+.
def TTMPType : GPRegTypeBase<"TTMP", "ttmp"> {
  let summary = "TTMP trap-temporary register type";
  let extraClassDeclaration = [{
    static constexpr RegisterKind kRegisterKind = RegisterKind::TTMP;
  }];
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
