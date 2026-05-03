//===- DependentOpInterface.h - Dependent Op Interface ----------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dependent operation interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_DEPENDENTOPINTERFACE_H
#define ASTER_INTERFACES_DEPENDENTOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#include "aster/Interfaces/DependentTokenTypeInterface.h.inc"

namespace mlir::aster {
/// Class for representing a future object. Note that futures are uniquely
/// identified by their token, and the value is optional.
struct Future {
  Future() = default;
  Future(Value token, bool isRead, Value value = nullptr)
      : token(token), value(value), isRead(isRead) {}

  /// Check if the future is valid.
  operator bool() const { return token != nullptr; }

  /// Check if the future is equal to another future.
  bool operator==(const Future &other) const { return token == other.token; }

  /// Create a future with a read effect.
  static Future read(Value token, Value value) {
    return Future(token, true, value);
  }
  /// Create a future with a write effect.
  static Future write(Value token, Value value = nullptr) {
    return Future(token, false, value);
  }

  /// Get the token associated with the future.
  Value getToken() const { return token; }
  /// Get the value associated with the future.
  Value getValue() const { return value; }
  /// Check if the future has a read effect.
  bool hasReadEffect() const { return isRead; }
  /// Check if the future has a write effect.
  bool hasWriteEffect() const { return !isRead; }

private:
  Value token;
  Value value;
  bool isRead = false;
};
} // namespace mlir::aster

#include "aster/Interfaces/DependentOpInterface.h.inc"

#endif // ASTER_INTERFACES_DEPENDENTOPINTERFACE_H
