//===- InstOpInterfaces.td - Inst Op Interfaces ------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines shared instruction operation interfaces for Aster.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_INSTOPINTERFACES_TD
#define ASTER_INTERFACES_INSTOPINTERFACES_TD

include "mlir/IR/OpBase.td"
include "aster/Interfaces/InstOpInterface.td"

//===----------------------------------------------------------------------===//
// Mov instruction interface
//===----------------------------------------------------------------------===//

def MovInstOpInterface : OpInterface<"MovInstOpInterface", [InstOpInterface]> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for move/copy-like instructions that transfer a value from one
    register to another without any computation.
  }];
  let methods = [
    InterfaceMethod<"Get the destination operand.",
      "::mlir::aster::Operand", "getDestOperand">,
    InterfaceMethod<"Get the destination result.",
      "::mlir::Value", "getDestResult">,
    InterfaceMethod<"Get the source operand.",
      "::mlir::aster::Operand", "getSrcOperand">
  ];
}

#endif // ASTER_INTERFACES_INSTOPINTERFACES_TD
