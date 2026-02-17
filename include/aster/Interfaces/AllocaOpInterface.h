//===- AllocaOpInterface.h - Alloca Op interface -----------------*- C++
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
// This file defines the aster alloca operation interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_ALLOCAOPINTERFACE_H
#define ASTER_INTERFACES_ALLOCAOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace aster {
class AllocaOpInterface;
namespace detail {
/// Verify the alloca operation.
LogicalResult verifyAllocaOpImpl(AllocaOpInterface op);
} // namespace detail
} // namespace aster
} // namespace mlir

#include "aster/Interfaces/AllocaOpInterface.h.inc"

#endif // ASTER_INTERFACES_ALLOCAOPINTERFACE_H
