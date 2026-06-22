//===- InstMethodUtils.h - helpers for generated inst methods --*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers referenced by amdgcn-tblgen-generated inst method bodies.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_INST_METHOD_UTILS_H
#define ASTER_DIALECT_AMDGCN_IR_INST_METHOD_UTILS_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/IR/InstImpl.h"
#include "aster/IR/ParsePrintUtils.h"
#include "mlir/IR/Value.h"
#include <type_traits>
#include <utility>

/// Helper to get either the type of a Value or return the type itself.
template <typename T,
          std::enable_if_t<std::is_base_of_v<mlir::Value, T>, int> = 0>
inline auto getTypeOrValue(T value) {
  using Type = decltype(value.getType());
  if (value == nullptr)
    return Type();
  return value.getType();
}
/// Helper to passthrough values that are not MLIR Values.
template <typename T,
          std::enable_if_t<!std::is_base_of_v<mlir::Value, T>, int> = 0>
inline T &&getTypeOrValue(T &&value) {
  return std::forward<T>(value);
}

#endif // ASTER_DIALECT_AMDGCN_IR_INST_METHOD_UTILS_H
