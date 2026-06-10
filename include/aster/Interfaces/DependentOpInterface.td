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

def BarrierOpInterface : OpInterface<"BarrierOpInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for barrier operations.
  }];
}

def FenceOpInterface : OpInterface<"FenceOpInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for operations that produce fence tokens. Wait ops and
    cross_wave_token_barrier implement this; consumers take fence_token via
    `FenceableOpInterface` to express single-thread operation
    ordering and enforce multi-thread synchronization.
  }];
  let methods = [
    InterfaceMethod<
      "Get the fence token produced by this op",
      "::mlir::Value", "getProducedFenceToken"
    >,
  ];
}

def FenceableOpInterface : OpInterface<"FenceableOpInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for operations that may consume an SSA fence token. This allows
    pinning the op after a FenceOpInterface producer to express single-thread
    operation ordering and enforce multi-thread synchronization.
  }];
  let methods = [
    InterfaceMethod<
      "Get the fence token consumed by this op.",
      "::mlir::Value", "getFenceToken"
    >,
    InterfaceMethod<
      "Get the mutable fence token operand.",
      "::mlir::MutableOperandRange", "getFenceTokenMutable"
    >,
  ];
  let extraSharedClassDeclaration = [{
    /// True if this op carries an SSA fence token operand.
    bool hasFenceToken() {
      return $_op.getFenceToken() != nullptr;
    }

    /// Drop the optional fence_token operand. No-op if absent.
    void eraseFence() {
      if (!hasFenceToken())
        return;
      $_op.getFenceTokenMutable().clear();
    }
  }];
}

def FutureOpInterface : OpInterface<"FutureOpInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for operations that produce futures.
  }];
  let methods = [
    InterfaceMethod<
      "Get the future object",
      "::mlir::aster::Future", "getFuture"
    >
  ];
}

#endif // ASTER_INTERFACES_DEPENDENTOPINTERFACE_TD
