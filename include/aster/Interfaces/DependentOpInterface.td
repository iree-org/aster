//===- DependentOpInterface.td - Dependent Op Interface ----*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dependent operation and token type interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_DEPENDENTOPINTERFACE_TD
#define ASTER_INTERFACES_DEPENDENTOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def TokenDependencyTypeInterface : TypeInterface<"TokenDependencyTypeInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for types that represent dependency tokens used for
    expressing ordering constraints between operations.
  }];
}

def DependentOpInterface : OpInterface<"DependentOpInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for operations that have dependency tokens for expressing
    ordering constraints between operations.
  }];
  let methods = [
    InterfaceMethod<
      "Get the dependency token operands",
      "::mlir::MutableOperandRange", "getDependenciesMutable"
    >,
    InterfaceMethod<
      "Get the output dependency token",
      "::mlir::Value", "getOutDependency"
    >
  ];
  let extraSharedClassDeclaration = [{
    /// Get the dependency token operands.
    ::mlir::ValueRange getDependencies() {
      return $_op.getDependenciesMutable().getAsOperandRange();
    }
  }];
}

#endif // ASTER_INTERFACES_DEPENDENTOPINTERFACE_TD
