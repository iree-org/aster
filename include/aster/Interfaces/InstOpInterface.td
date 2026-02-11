//===- InstOpInterface.td - InstOp interface ---------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the instruction operation interface for Aster.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_INSTOPINTERFACE_TD
#define ASTER_INTERFACES_INSTOPINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// InstOp Interface
//===----------------------------------------------------------------------===//

def InstOpInterface : OpInterface<"InstOpInterface"> {
  let description = [{
    Interface for instruction operations. This interface expects that for
    each output operand there is a corresponding result, and that they appear in
    the same order.
  }];
  let cppNamespace = "::mlir::aster";
  let methods = [
    InterfaceMethod<
      "Get the instruction output operands",
      "::mlir::OperandRange", "getInstOuts"
    >,
    InterfaceMethod<
      "Get the instruction input operands",
      "::mlir::OperandRange", "getInstIns"
    >,
    InterfaceMethod<
      "Get the instruction results",
      "::mlir::ResultRange", "getInstResults", (ins), [{}], [{
        return $_op->getResults();
      }]
    >,
    InterfaceMethod<
      "Returns true if the instruction has been register allocated",
      "bool", "hasPureValueSemantics", (ins), [{}], [{
        return ::mlir::aster::detail::hasPureValueSemanticsImpl($_op);
      }]
    >,
    InterfaceMethod<
      "Returns true if the instruction has DPS-like semantics",
      "bool", "isDPSInstruction", (ins), [{}], [{
        // TODO: Remove this default implementation when all instructions have proper
        // overrides.
        return true;
      }]
    >,
    InterfaceMethod< [{
        Returns a clone of the instruction with the provided instruction operands and results,
        if the result types are not provided, the clone will deduce the result types from the
        instruction operands. Returns nullptr on failure.
      }],  "::mlir::aster::InstOpInterface", "cloneInst", (ins
          "::mlir::OpBuilder&":$builder, "::mlir::ValueRange":$outs,
          "::mlir::ValueRange":$ins, "std::optional<::mlir::TypeRange>":$resultTypes
        ), [{}], [{
        return ::mlir::aster::detail::cloneInstOpImpl($_op, builder, outs, ins, resultTypes);
      }]
    >,
  ];
  let verify = [{
    return ::mlir::aster::detail::verifyInstImpl(cast<ConcreteOp>($_op));
  }];
}

#endif // ASTER_INTERFACES_INSTOPINTERFACE_TD
