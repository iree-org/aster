//===- AllocaOpInterface.td - Alloca Op Interface --------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the alloca operation interface for Aster.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_ALLOCAOPINTERFACE_TD
#define ASTER_INTERFACES_ALLOCAOPINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AllocaOp Interface
//===----------------------------------------------------------------------===//

def AllocaOpInterface : OpInterface<"AllocaOpInterface"> {
  let description = [{
    Interface for operations that allocate memory (alloca-like operations).
  }];
  let cppNamespace = "::mlir::aster";
  let methods = [
    InterfaceMethod<
      "Get the alloca result",
      "::mlir::Value", "getAlloca"
    >
  ];
  let verify = [{
    return ::mlir::aster::detail::verifyAllocaOpImpl(cast<ConcreteOp>($_op));
  }];
}

#endif // ASTER_INTERFACES_ALLOCAOPINTERFACE_TD
