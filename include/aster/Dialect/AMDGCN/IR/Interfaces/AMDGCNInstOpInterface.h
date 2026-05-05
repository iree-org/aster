//===- AMDGCNInstOpInterface.h - AMDGCN instruction op interface -*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCNInstOpInterface for AMDGCN instruction
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_INTERFACES_AMDGCNINSTOPINTERFACE_H
#define ASTER_DIALECT_AMDGCN_IR_INTERFACES_AMDGCNINSTOPINTERFACE_H

#include "aster/Interfaces/InstOpInterface.h"
#include "mlir/IR/Operation.h"

namespace mlir::aster::amdgcn {
enum class OpCode : uint64_t;
enum class InstProp : uint32_t;
enum class ISAVersion : uint32_t;
} // namespace mlir::aster::amdgcn

#include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.h.inc"

#endif // ASTER_DIALECT_AMDGCN_IR_INTERFACES_AMDGCNINSTOPINTERFACE_H
