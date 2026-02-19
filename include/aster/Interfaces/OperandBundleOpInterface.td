//===- OperandBundleOpInterface.td - OperandBundleOp Iface -*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operand bundle operation interface for Aster.
// Operations implementing this interface group operands into a single result.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_OPERANDBUNDLEOPINTERFACE_TD
#define ASTER_INTERFACES_OPERANDBUNDLEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// OperandBundleOp Interface
//===----------------------------------------------------------------------===//

def OperandBundleOpInterface : OpInterface<"OperandBundleOpInterface"> {
  let description = [{
    Interface for operations that group operands into a single result.
    The operation has exactly one result and one or more operands.
  }];
  let cppNamespace = "::mlir::aster";
  let methods = [
    InterfaceMethod<[{
      Unpack the operand bundle into the values that form the single result.
    }], "::mlir::ValueRange", "unpackBundle", (ins)
    >
  ];
  let verify = [{
    return ::mlir::aster::detail::verifyOperandBundleImpl(cast<ConcreteOp>($_op));
  }];
}

#endif // ASTER_INTERFACES_OPERANDBUNDLEOPINTERFACE_TD
