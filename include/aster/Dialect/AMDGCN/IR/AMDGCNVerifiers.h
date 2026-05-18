//===- AMDGCNVerifiers.h - AMDGCN verifier attributes ---------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN verifier implementation.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_VERIFIERS_H
#define ASTER_DIALECT_AMDGCN_IR_VERIFIERS_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::aster::amdgcn {
/// Verify ISA support for operations in a region.
/// If isas is empty, no AMDGCNInstOpInterface operations are allowed.
/// If isas is non-empty, all AMDGCNInstOpInterface operations must be valid
/// for ALL the listed ISA versions.
LogicalResult
verifyISAsSupportImpl(Region &region, ArrayRef<ISAVersion> isas,
                      function_ref<InFlightDiagnostic()> emitError);
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_IR_VERIFIERS_H
