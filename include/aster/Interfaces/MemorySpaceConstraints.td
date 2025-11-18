//===- MemorySpaceConstraints.td - Memory Space Constraints ----*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the memory space constraints attribute interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_MEMORYSPACECONSTRAINTS_TD
#define ASTER_INTERFACES_MEMORYSPACECONSTRAINTS_TD

include "mlir/IR/OpBase.td"

def MemorySpaceConstraintsAttrInterface
    : AttrInterface<"MemorySpaceConstraintsAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for memory space attributes that provide constraints on
    memory operations, such as supported operation widths.
  }];
  let methods = [
    InterfaceMethod<
      "Get the supported operation widths in bits for this memory space",
      "::mlir::LogicalResult", "getSupportedOpWidths",
      (ins
        "::mlir::Type":$type,
        "::mlir::Value":$addr,
        "::mlir::Value":$offset,
        "::mlir::Value":$const_offset,
        "bool":$isRead,
        "::llvm::SmallVectorImpl<int32_t>&":$widths)
    >
  ];
}

#endif // ASTER_INTERFACES_MEMORYSPACECONSTRAINTS_TD
