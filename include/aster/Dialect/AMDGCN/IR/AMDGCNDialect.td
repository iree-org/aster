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
include "aster/IR/InstBase.td"
include "aster/IR/Inst.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.td"
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
  let extraClassDeclaration = [{
    /// Maximum number of SGPR registers supported by this dialect
    static constexpr int16_t maxSGPRRegisters = 102;
    /// Maximum number of VGPR registers supported by this dialect
    static constexpr int16_t maxVGPRRegisters = 256;
    /// Maximum number of AGPR registers supported by this dialect
    static constexpr int16_t maxAGPRRegisters = 256;

    /// Parse an attribute registered to this dialect.
    ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                        ::mlir::Type type) const override;

    /// Print an attribute registered to this dialect.
    void printAttribute(::mlir::Attribute attr,
        ::mlir::DialectAsmPrinter &os) const override;

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

/// AMDGCN instruction property.
class InstProp<string sym> : PrettyEnumCaseInfo<sym>;

/// AMDGCN ISA version spec.
class ISAVersion<string sym, string id = ""> : EnumCaseInfo<sym, id>;

/// AMDGCN target.
class Target<string name, ISAVersion version> : EnumCaseInfo<name, ""> {
  ISAVersion isaVersion = version;
}

/// AMDGCN instruction constraints.
class InstConstraint<list<Constraint> _constraints = []> {
  // The list of constraints that have to be satisfied.
  list<Constraint> constraints = _constraints;
}

class AMDInstOpCode<string sym, string id = ""> : EnumCaseInfo<sym, id>;

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
// AMDGCN InstOp base class
//===----------------------------------------------------------------------===//

class AMDGCN_InstOp<string mnemonic, list<Trait> traits = []>
    : InstOp<AMDGCN_Dialect, mnemonic,
                    traits # [AMDGCNInstOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn::inst";
}

#endif // ASTER_AMDGCN_DIALECT
