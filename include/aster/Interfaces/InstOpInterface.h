//===- InstOpInterface.h - InstOp interface ---------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the aster instruction operation interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_INSTOPINTERFACE_H
#define ASTER_INTERFACES_INSTOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::aster {
class InstOpInterface;
namespace detail {
/// Verify the instruction operation.
LogicalResult verifyInstImpl(InstOpInterface op);
/// Returns true if the instruction has been register allocated.
bool isRegAllocatedImpl(InstOpInterface op);
struct InstAttrStorage;
} // namespace detail
} // namespace mlir::aster

#include "aster/Interfaces/InstOpInterface.h.inc"

#endif // ASTER_INTERFACES_INSTOPINTERFACE_H
