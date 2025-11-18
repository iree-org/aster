//===- ResourceInterfaces.h - Resource Interfaces ---------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines resource type and allocation operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_RESOURCEINTERFACES_H
#define ASTER_INTERFACES_RESOURCEINTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::aster {
using Resource = SideEffects::Resource;
} // namespace mlir::aster

#include "aster/Interfaces/ResourceOpInterfaces.h.inc"
#include "aster/Interfaces/ResourceTypeInterfaces.h.inc"

#endif // ASTER_INTERFACES_RESOURCEINTERFACES_H
