//===- LivenessOpInterface.td - Liveness Op Interface ------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the liveness operation interface for Aster.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_LIVENESSOPINTERFACE_TD
#define ASTER_INTERFACES_LIVENESSOPINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LivenessOp Interface
//===----------------------------------------------------------------------===//

def LivenessOpInterface : OpInterface<"LivenessOpInterface"> {
  let description = [{
    Interface for operations that require special handling when computing
    liveness information.
    Operations implementing this interface specify which values they use (become
    live in backward liveness) and which values they define (become dead in
    backward liveness).
  }];
  let cppNamespace = "::mlir::aster";
  let methods = [
    InterfaceMethod<[{
      Computes liveness transfer function for the operation. The callback
      functions are called to insert and remove live values from the liveness
      analysis. Additionally, the isLiveCallback can be called to check if a
      value is live. Returns failure if the liveness transfer function cannot be
      computed.
    }], "::mlir::LogicalResult", "livenessTransferFunction",
    (ins "::mlir::aster::LivenessCallback":$insertCallback,
          "::mlir::aster::LivenessCallback":$removeCallback,
          "::mlir::aster::IsLiveCallback":$isLiveCallback)
    >
  ];
}

#endif // ASTER_INTERFACES_LIVENESSOPINTERFACE_TD
