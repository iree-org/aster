//===- AMDGCNAttrs.h - AMDGCN Attributes ------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_AMDGCNATTRS_H
#define ASTER_DIALECT_AMDGCN_IR_AMDGCNATTRS_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.h"
#include "aster/Interfaces/MemorySpaceConstraints.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/IR/Attributes.h"

namespace mlir {
class DataLayout;
namespace aster::amdgcn {
class InstAttr;
namespace detail {
struct InstAttrStorage;
} // namespace detail
} // namespace aster::amdgcn
} // namespace mlir

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h.inc"

#endif // ASTER_DIALECT_AMDGCN_IR_AMDGCNATTRS_H
