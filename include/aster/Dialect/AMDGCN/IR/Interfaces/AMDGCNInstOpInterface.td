//===- AMDGCNInstOpInterface.td - InstOp interface ---------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_INSTOPINTERFACE
#define ASTER_AMDGCN_INSTOPINTERFACE

include "mlir/IR/OpBase.td"
include "aster/Interfaces/InstOpInterface.td"

//===----------------------------------------------------------------------===//
// AMDGCNInstOp Interface
//===----------------------------------------------------------------------===//

def AMDGCNInstOpInterface :
    OpInterface<"AMDGCNInstOpInterface", [InstOpInterface]> {
  let description = [{
    Interface for AMDGCN instruction operations. This interface expects that for
    each output operand there is a corresponding result, and that they appear in
    the same order.
  }];
  let cppNamespace = "::mlir::aster::amdgcn";
  let methods = [
    InterfaceMethod<
      "Returns the opcode of the instruction",
      "::mlir::aster::amdgcn::InstAttr", "getOpcodeAttr"
    >,
    InterfaceMethod<
      "Temporary method, TODO: remove",
      "bool", "isNewInst", (ins), [{}], [{
        return false;
      }]
    >,
    InterfaceMethod<
      "Returns the opcode of the instruction",
      "::mlir::aster::amdgcn::OpCode", "getOpCode", (ins), [{}], [{
        const ::mlir::aster::amdgcn::InstMetadata* md = $_op.getInstMetadata();
        return md ? md->getOpCode() : ::mlir::aster::amdgcn::OpCode::Invalid;
      }]
    >,
    InterfaceMethod<
      "Returns whether the instruction has the given inst trait",
      "bool", "hasInstProps", (ins "::llvm::ArrayRef<InstProp>":$props, "bool":$all), [{}], [{
        ::mlir::aster::amdgcn::InstAttr attr = $_op.getOpcodeAttr();
        if (!attr)
          return false;
        const ::mlir::aster::amdgcn::InstMetadata *md = attr.getMetadata();
        if (!md)
          return false;
        return all ? md->hasProps(props) : md->hasAnyProps(props);
      }]
    >,
    InterfaceMethod<
      "Returns the available ISA versions of the instruction",
      "::llvm::ArrayRef<::mlir::aster::amdgcn::ISAVersion>", "getISAVersions", (ins), [{}], [{
        const ::mlir::aster::amdgcn::InstMetadata* md = $_op.getInstMetadata();
        return md ? md->getISAVersions() : ::llvm::ArrayRef<::mlir::aster::amdgcn::ISAVersion>();
      }]
    >
  ];
  let extraSharedClassDeclaration = [{
    /// Get the instruction metadata for the instruction or nullptr if the
    /// instruction has no opcode attribute.
    const ::mlir::aster::amdgcn::InstMetadata* getInstMetadata() {
      ::mlir::aster::amdgcn::InstAttr attr = $_op.getOpcodeAttr();
      return attr ? attr.getMetadata() : nullptr;
    }

    bool hasProp(::mlir::aster::amdgcn::InstProp prop) {
      return $_op.hasInstProps({prop}, true);
    }

    bool hasProps(::llvm::ArrayRef<::mlir::aster::amdgcn::InstProp> props) {
      return $_op.hasInstProps(props, true);
    }

    bool hasAnyProps(::llvm::ArrayRef<::mlir::aster::amdgcn::InstProp> props) {
      return $_op.hasInstProps(props, false);
    }

    /// Returns true if the instruction supports the given ISA version.
    /// An instruction with no ISA version list is assumed to support all.
    bool supportsISA(::mlir::aster::amdgcn::ISAVersion ver) {
      llvm::ArrayRef<::mlir::aster::amdgcn::ISAVersion> isas = $_op.getISAVersions();
      return llvm::is_contained(isas, ver);
    }
  }];
}

#endif // ASTER_AMDGCN_INSTOPINTERFACE
