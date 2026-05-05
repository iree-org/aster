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
      "::mlir::aster::amdgcn::OpCode", "getOpCode"
    >,
    InterfaceMethod<
      "Returns whether the instruction has the given inst trait",
      "bool", "hasInstProps", (ins "::llvm::ArrayRef<::mlir::aster::amdgcn::InstProp>":$props, "bool":$all)
    >,
    InterfaceMethod<
      "Returns the available ISA versions of the instruction",
      "::llvm::ArrayRef<::mlir::aster::amdgcn::ISAVersion>", "getISAVersions"
    >
  ];
  let extraSharedClassDeclaration = [{
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
