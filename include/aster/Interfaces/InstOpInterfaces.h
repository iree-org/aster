//===- InstOpInterfaces.h - Inst Op Interfaces ------------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines shared instruction operation interfaces for Aster.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_INSTOPINTERFACES_H
#define ASTER_INTERFACES_INSTOPINTERFACES_H

#include "aster/IR/Operand.h"
#include "aster/Interfaces/InstOpInterface.h"

#include "aster/Interfaces/InstOpInterfaces.h.inc"
#include "mlir/IR/PatternMatch.h"

namespace mlir::aster::detail {
/// Canonicalization pattern for MovInstOpInterface, if the source and
/// destination are the same register, the instruction can be removed.
LogicalResult canonicalizeMovInstImpl(MovInstOpInterface mov,
                                      RewriterBase &rewriter);
} // namespace mlir::aster::detail

#endif // ASTER_INTERFACES_INSTOPINTERFACES_H
