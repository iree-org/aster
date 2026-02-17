//===- LivenessOpInterface.h - Liveness Op interface ------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the aster liveness operation interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_LIVENESSOPINTERFACE_H
#define ASTER_INTERFACES_LIVENESSOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::aster {
class LivenessOpInterface;

/// A callback function that is called to add live or remove dead values from
/// the liveness analysis.
using LivenessCallback = llvm::function_ref<void(ValueRange)>;

/// A callback function that is called to check if a value is live.
using IsLiveCallback = llvm::function_ref<bool(Value)>;
} // namespace mlir::aster

#include "aster/Interfaces/LivenessOpInterface.h.inc"

#endif // ASTER_INTERFACES_LIVENESSOPINTERFACE_H
