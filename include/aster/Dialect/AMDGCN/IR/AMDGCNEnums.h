//===- AMDGCNEnums.h - AMDGCN enums -----------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN enums.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_AMDGCNENUMS_H
#define ASTER_DIALECT_AMDGCN_AMDGCNENUMS_H

#include "mlir/IR/BuiltinAttributes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h.inc"

namespace mlir::aster::amdgcn {
/// Helper to check if the address space kind matches any of the given kinds.
inline bool isAddressSpaceOf(AddressSpaceKind value,
                             ArrayRef<AddressSpaceKind> kinds) {
  return llvm::is_contained(kinds, value);
}
/// Helper to check if the access kind matches any of the given kinds.
inline bool isAccessKindOf(AccessKind value, ArrayRef<AccessKind> kinds) {
  return llvm::is_contained(kinds, value);
}
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_AMDGCNENUMS_H
