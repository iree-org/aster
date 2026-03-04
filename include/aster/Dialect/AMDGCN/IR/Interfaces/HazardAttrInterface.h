//===- HazardAttrInterface.h - Hazard attribute interface -------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HazardRaiserAttrInterface and
// HazardCheckerAttrInterface for hazard attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_INTERFACES_HAZARDATTRINTERFACE_H
#define ASTER_DIALECT_AMDGCN_IR_INTERFACES_HAZARDATTRINTERFACE_H

#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.h"
#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/IR/Attributes.h"

namespace mlir::aster::amdgcn {
struct Hazard;
} // namespace mlir::aster::amdgcn

#include "aster/Dialect/AMDGCN/IR/Interfaces/HazardAttrInterface.h.inc"

#endif // ASTER_DIALECT_AMDGCN_IR_INTERFACES_HAZARDATTRINTERFACE_H
