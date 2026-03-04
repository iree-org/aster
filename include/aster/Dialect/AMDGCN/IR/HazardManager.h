//===- HazardManager.h - AMDGCN hazard manager ------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HazardManager for AMDGCN hazard detection.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_HAZARDMANAGER_H
#define ASTER_DIALECT_AMDGCN_IR_HAZARDMANAGER_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/Hazards.h"
#include "aster/Dialect/AMDGCN/IR/Interfaces/HazardAttrInterface.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace aster::amdgcn {

/// A hazard manager is responsible for computing the hazards for a given
/// operation.
struct HazardManager {
  /// Create a new hazard manager for the given top operation.
  HazardManager(Operation *topOp) : topOp(topOp) {
    assert(topOp != nullptr && "topOp cannot be nullptr");
  }

  /// Populate hazard attributes for the given ISA version.
  void populateHazardsFor(ISAVersion version);

  /// Get the hazards for the given operation. Returns failure if the hazards
  /// cannot be computed.
  LogicalResult getHazards(AMDGCNInstOpInterface instOp,
                           SmallVectorImpl<Hazard> &hazards);

  /// Get the top operation of the hazard manager.
  Operation *getTopOp() const { return topOp; }

private:
  Operation *topOp;

  /// Get the hazard raisers for the given ISA version.
  void getHazardRaisersFor(
      ISAVersion version,
      SmallVectorImpl<HazardRaiserAttrInterface> &hazardRaisers);

  /// The hazard raisers participating in the hazard analysis.
  SmallVector<std::pair<OpCode, HazardRaiserAttrInterface>> hazardAttrs;

  /// A map from an opcode to the [begin, end) indices in `hazardAttrs` that
  /// match the opcode.
  DenseMap<OpCode, std::pair<int32_t, int32_t>> opcodeToHazardAttrs;
};

} // namespace aster::amdgcn
} // namespace mlir

#endif // ASTER_DIALECT_AMDGCN_IR_HAZARDMANAGER_H
