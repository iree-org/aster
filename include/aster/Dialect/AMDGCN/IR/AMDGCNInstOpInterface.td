//===- AMDGCNInstOpInterface.td - InstOp interface ---------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INST_OP_INTERFACE
#define AMDGCN_INST_OP_INTERFACE

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
      // TODO: refine the Attribute type.
      "Returns the opcode of the instruction",
      "::mlir::aster::amdgcn::InstAttr", "getOpcodeAttr"
    >
  ];
}

#endif // AMDGCN_ASSEMBLY_INTERFACE
