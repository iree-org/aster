//===- AMDGCN.td - AMDGCN Dialect --------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the amdgcn dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_DIALECT
#define ASTER_AMDGCN_DIALECT

include "mlir/IR/OpBase.td"
include "aster/Support/EnumBase.td"
include "aster/Support/Utils.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AMDGCN Dialect
//===----------------------------------------------------------------------===//

def AMDGCN_Dialect : Dialect {
  let name = "amdgcn";
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    AMDGCN dialect.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
  let extraClassDeclaration = [{
    /// Maximum number of SGPR registers supported by this dialect
    static constexpr int16_t maxSGPRRegisters = 102;
    /// Maximum number of VGPR registers supported by this dialect
    static constexpr int16_t maxVGPRRegisters = 256;
    /// Maximum number of AGPR registers supported by this dialect
    static constexpr int16_t maxAGPRRegisters = 256;

  private:
    /// Initialize the dialect's attributes.
    void initializeAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base AMDGCN Op class
//===----------------------------------------------------------------------===//

class AMDGCN_Op<string mnemonic, list<Trait> traits = []>
    : Op<AMDGCN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Instruction infrastructure definitions
//===----------------------------------------------------------------------===//

/// AMDGCN ISA version spec.
class ISAVersion<string sym, string id = ""> : EnumCaseInfo<sym, id>;

/// AMDGCN instruction class.
class InstClass<string sym, string id = ""> : EnumCaseInfo<sym, id>;

/// AMDGCN target.
class Target<string name, ISAVersion version> : EnumCaseInfo<name, ""> {
  ISAVersion isaVersion = version;
}

/// AMDGCN argument assembly format.
class AsmArgFormat {
  // Assembly parser code.
  // Example: `$_parser.parseOperand($_self);`
  code asmParser = ?;
  // Assembly printer code.
  // Example: `$_printer.printOperand($_self);`
  code asmPrinter = ?;
}

/// Python attribute with a builder.
/// This is used to generate Python bindings for attributes that have
/// simple builders.
class PyAttr {
  OpBuilder builder = ?;
}

/// AMDGCN instruction constraints.
class InstConstraint<list<Constraint> _constraints = []> {
  // The list of constraints that have to be satisfied.
  list<Constraint> constraints = _constraints;
}

/// AMDGCN instruction op class.
class InstOp<string mnemonic, list<Trait> traits = []>
  : AMDGCN_Op<mnemonic, traits> {
  let cppNamespace = "::mlir::aster::amdgcn::inst";
  // Whether to generate the instruction verifier using amdgcn-tblgen.
  bit genInstVerifier = 1;
  let hasVerifier = !if(genInstVerifier, 1, 0);
}

/// AMDGCN instruction assembly variant.
class AsmVariant<Pred pred, code format> {
  // The predicate that determines when this variant applies.
  Constraint predicate = Constraint<pred>;
  // The assembly format string.
  // The only acceptable tokens that can appear are:
  // - `$<identifier>`: represents an argument named <identifier>
  // - `,`: represents a comma
  // - `<identifier>`: represents a literal identifier
  // identifier:= `[a-zA-Z_][a-zA-Z0-9_]*`
  code asmFormat = format;
}

/// AMDGCN instruction definition.
class AMDInst<string _mnemonic, string sym, InstOp _instOp, list<ISAVersion> _isa>
    : EnumCaseInfo<sym, ""> {
  // The summary description of the instruction.
  code summary = "AMDGCN instruction `" # _mnemonic # "`.";
  code description = summary;
  // The mnemonic of the instruction.
  string mnemonic = _mnemonic;
  // The AMDGCN ISA this instruction is valid for.
  list<ISAVersion> isa = _isa;
  // The assembly format variants for this instruction.
  list<AsmVariant> asmFormat = ?;
  // The C++ namespace where the instruction op is defined.
  string cppNamespace = "::mlir::aster::amdgcn";
  // The instruction op class.
  InstOp instOp = _instOp;
  // The instruction constraints.
  dag constraints = (ins);
  // The op builders for C++ and Python.
  // These are used to generate builder methods for the instruction.
  OpBuilder cppBuilder = ?;
  // If specified this generates a python function to build the instruction.
  OpBuilder pythonBuilder = ?;
  // Extra optional class declaration/definition.
  code extraClassDeclaration = "";
  code extraClassDefinition = "";
  // Whether the instruction has an initializer.
  bit hasInit = 0;
}

//===----------------------------------------------------------------------===//
// Base AMDGCN Attr, Type, and Enum classes
//===----------------------------------------------------------------------===//

class AMDGCN_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AMDGCN_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class AMDGCN_EnumAttr<string name, string attrMnemonic, EnumInfo enumInfo, list<Trait> traits = []>
    : EnumAttr<AMDGCN_Dialect, enumInfo, name, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` $value `>`";
}

class AMDGCN_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AMDGCN_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Inst Op Helpers
//===----------------------------------------------------------------------===//

// DS instruction op.
class DS_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("ds.", mnemonic), traits> {
}

// FLAT instruction op.
class FLAT_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("flat.", mnemonic), traits> {
}

// MTBUF instruction op.
class MTBUF_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("mtbuf.", mnemonic), traits> {
}

// MUBUF instruction op.
class MUBUF_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("mubuf.", mnemonic), traits> {
}

// SMEM instruction op.
class SMEM_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("smem.", mnemonic), traits> {
}

// SOP1 instruction op.
class SOP1_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("sop1.", mnemonic), traits> {
}

// SOP2 instruction op.
class SOP2_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("sop2.", mnemonic), traits> {
}

// SOPC instruction op.
class SOPC_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("sopc.", mnemonic), traits> {
}

// SOPK instruction op.
class SOPK_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("sopk.", mnemonic), traits> {
}

// SOPP instruction op.
class SOPP_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("sopp.", mnemonic), traits> {
}

// VOP1 instruction op.
class VOP1_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("vop1.", mnemonic), traits> {
}

// VOP2 instruction op.
class VOP2_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("vop2.", mnemonic), traits> {
}

// VOP3 instruction op.
class VOP3_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("vop3.", mnemonic), traits> {
}

// VOP3P instruction op.
class VOP3P_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("vop3p.", mnemonic), traits> {
}

// VOPC instruction op.
class VOPC_Op<string mnemonic, list<Trait> traits = []>
  : InstOp<!strconcat("vopc.", mnemonic), traits> {
}

//===----------------------------------------------------------------------===//
// Base traits
//===----------------------------------------------------------------------===//

// Note: https://discourse.llvm.org/t/question-about-memoryeffectsopinterface/88879
// MemoryEffectsOpInterface is the "generic" name for effects that prevent DCE.
// We generally want that because of WYSIWYG, as a consequence, need to be very
// explicit about resources.
def AMDGCNOpTraits : TraitList<[
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]>;

def AMDGCNInstOpTrait : NativeOpTrait<"InstOpTrait"> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let dependentTraits = [ConditionallySpeculatable, MemoryEffectsOpInterface];
  let extraConcreteClassDeclaration = [{
    /// Returns the speculatability of the operation.
    Speculation::Speculatability getSpeculatability() {
      return aster::detail::getInstSpeculatabilityImpl(*this);
    }
    /// Returns the memory effects of the operation.
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
            &effects) {
      return aster::detail::getInstEffectsImpl(*this, effects);
    }
  }];
}

// Note: https://discourse.llvm.org/t/question-about-memoryeffectsopinterface/88879
// MemoryEffectsOpInterface is the "generic" name for effects that prevent DCE.
// We generally want that because of WYSIWYG, as a consequence, need to be very
// explicit about resources.
def AMDGCNInstTraits : TraitList<[
    ConditionallySpeculatable,
    MemoryEffectsOpInterface,
    AMDGCNInstOpTrait
  ]>;

//===----------------------------------------------------------------------===//
// Memory Resources
//===----------------------------------------------------------------------===//

def GlobalMemoryResource : Resource<"::mlir::aster::amdgcn::GlobalMemoryResource">;
def LDSMemoryResource : Resource<"::mlir::aster::amdgcn::LDSMemoryResource">;

//===----------------------------------------------------------------------===//
// Memory Effect Traits
//===----------------------------------------------------------------------===//

class AMDGCNMemoryReadInstOpTraits<Resource memoryResource> : NativeOpTrait<"InstOpTrait"> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let dependentTraits = [ConditionallySpeculatable, MemoryEffectsOpInterface];
  let extraConcreteClassDeclaration = [{
    /// Returns the speculatability of the operation.
    Speculation::Speculatability getSpeculatability() {
      return aster::detail::getInstSpeculatabilityImpl(*this);
    }
    /// Returns the memory effects of the operation.
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
            &effects) {
      aster::detail::getInstEffectsImpl(*this, effects);
      // Add explicit memory read effect on addr register even if not allocated.
      if (getAddr()) {
        MutableOperandRange addrRange = getAddrMutable();
        for (OpOperand &operand : addrRange) {
          effects.emplace_back(MemoryEffects::Read::get(), &operand);
        }
      }
      // Add memory resource effects for the proper memory resource.
      effects.emplace_back(MemoryEffects::Read::get(), }] # memoryResource.name
      # [{::get());
    }
  }];
}

class AMDGCNMemoryReadInstTraits<Resource memoryResource> : TraitList<[
    ConditionallySpeculatable,
    MemoryEffectsOpInterface,
    // Note: cannot just use because we also need to dynamically derive the
    // precise resource in the case of registers and getEffects is already
    // defined.
    // MemoryEffects<[MemRead<memoryResource>]>,
    AMDGCNMemoryReadInstOpTraits<memoryResource>
  ]>;

class AMDGCNMemoryWriteInstOpTraits<Resource memoryResource> : NativeOpTrait<"InstOpTrait"> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let dependentTraits = [ConditionallySpeculatable, MemoryEffectsOpInterface];
  let extraConcreteClassDeclaration = [{
    /// Returns the speculatability of the operation.
    Speculation::Speculatability getSpeculatability() {
      return aster::detail::getInstSpeculatabilityImpl(*this);
    }
    /// Returns the memory effects of the operation.
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
            &effects) {
      aster::detail::getInstEffectsImpl(*this, effects);
      // Add explicit memory write effect on addr register even if not allocated.
      MutableOperandRange addrRange = getAddrMutable();
      for (OpOperand &operand : addrRange) {
        effects.emplace_back(MemoryEffects::Write::get(), &operand);
      }
      // Add memory resource effects for the proper memory resource.
      effects.emplace_back(MemoryEffects::Write::get(), }] # memoryResource.name
      # [{::get());
    }
  }];
}

class AMDGCNMemoryWriteInstTraits<Resource memoryResource> : TraitList<[
    ConditionallySpeculatable,
    MemoryEffectsOpInterface,
    // Note: cannot just use because we also need to dynamically derive the
    // precise resource in the case of registers and getEffects is already
    // defined.
    // MemoryEffects<[MemWrite<memoryResource>]>,
    AMDGCNMemoryWriteInstOpTraits<memoryResource>
  ]>;

#endif // ASTER_AMDGCN_DIALECT
