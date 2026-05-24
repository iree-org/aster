// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ConvertMemSpaceToAMDGCN.cpp - integer memspace -> amdgcn addr_space ===//
//
// Converts AIR-style integer memory spaces on memref types to AMDGCN address
// space attributes:
//
//   null / 0  →  #amdgcn.addr_space<global, read_write>
//   2 (L1)   →  #amdgcn.addr_space<local, read_write>
//
// This pass bridges the gap between the upstream AIR pipeline (which uses
// integer memory spaces) and aster's AMDGCN decomposition passes (which
// require #amdgcn.addr_space attributes for pointer generation).
//
// Run after air-to-amdgcn and before convert-to-amdgcn-library-calls.
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::aster::amdgcn;

namespace {

/// Map an integer memory space to an AMDGCN AddressSpaceAttr.
/// Returns nullptr if no mapping is needed (e.g., already an AMDGCN attr).
static AddressSpaceAttr mapMemorySpace(Attribute memSpace,
                                       MLIRContext *ctx) {
  // null or integer 0 → global
  if (!memSpace) {
    return AddressSpaceAttr::get(ctx, AddressSpaceKind::Global,
                                AccessKind::ReadWrite);
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(memSpace)) {
    unsigned space = intAttr.getInt();
    switch (space) {
    case 0:
      return AddressSpaceAttr::get(ctx, AddressSpaceKind::Global,
                                   AccessKind::ReadWrite);
    case 2:
      return AddressSpaceAttr::get(ctx, AddressSpaceKind::Local,
                                   AccessKind::ReadWrite);
    default:
      // Unknown integer space — map to global as fallback.
      return AddressSpaceAttr::get(ctx, AddressSpaceKind::Global,
                                   AccessKind::ReadWrite);
    }
  }
  // Already a non-integer attribute (e.g., #amdgcn.addr_space) — no change.
  return {};
}

/// Convert a MemRefType's memory space if needed.
static MemRefType convertMemRefType(MemRefType ty, MLIRContext *ctx) {
  auto newSpace = mapMemorySpace(ty.getMemorySpace(), ctx);
  if (!newSpace)
    return ty;
  // Use the AffineMap overload since MemRefLayoutAttrInterface overload
  // doesn't exist for some MLIR versions.
  Attribute spaceAttr = newSpace;
  return MemRefType::get(ty.getShape(), ty.getElementType(),
                         ty.getLayout().getAffineMap(), spaceAttr);
}

struct ConvertMemSpaceToAMDGCN
    : public PassWrapper<ConvertMemSpaceToAMDGCN,
                         InterfacePass<aster::ModuleOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMemSpaceToAMDGCN)
  StringRef getArgument() const override {
    return "convert-memspace-to-amdgcn";
  }
  StringRef getDescription() const override {
    return "Convert integer memory spaces to #amdgcn.addr_space attributes";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDGCNDialect>();
  }

  void runOnOperation() override {
    Operation *moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Collect all func.func ops to update.
    SmallVector<func::FuncOp> funcs;
    moduleOp->walk([&](func::FuncOp f) { funcs.push_back(f); });

    for (auto funcOp : funcs) {
      // Update function signature.
      auto funcTy = funcOp.getFunctionType();
      bool changed = false;

      SmallVector<Type> newInputs;
      for (auto ty : funcTy.getInputs()) {
        if (auto mrTy = dyn_cast<MemRefType>(ty)) {
          auto newTy = convertMemRefType(mrTy, ctx);
          newInputs.push_back(newTy);
          if (newTy != mrTy)
            changed = true;
        } else {
          newInputs.push_back(ty);
        }
      }

      SmallVector<Type> newResults;
      for (auto ty : funcTy.getResults()) {
        if (auto mrTy = dyn_cast<MemRefType>(ty)) {
          auto newTy = convertMemRefType(mrTy, ctx);
          newResults.push_back(newTy);
          if (newTy != mrTy)
            changed = true;
        } else {
          newResults.push_back(ty);
        }
      }

      if (changed) {
        funcOp.setType(FunctionType::get(ctx, newInputs, newResults));
        // Update block argument types.
        if (!funcOp.empty()) {
          Block &entry = funcOp.front();
          for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
            if (auto mrTy = dyn_cast<MemRefType>(entry.getArgument(i).getType())) {
              auto newTy = convertMemRefType(mrTy, ctx);
              if (newTy != mrTy)
                entry.getArgument(i).setType(newTy);
            }
          }
        }
      }

      // Walk all ops inside the function and update memref types.
      funcOp->walk([&](Operation *op) {
        // Update operand types (handled transitively via result types).
        // Update result types.
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
          auto ty = op->getResult(i).getType();
          if (auto mrTy = dyn_cast<MemRefType>(ty)) {
            auto newTy = convertMemRefType(mrTy, ctx);
            if (newTy != mrTy)
              op->getResult(i).setType(newTy);
          }
        }
        // Update block argument types in regions.
        for (auto &region : op->getRegions()) {
          for (auto &block : region) {
            for (auto arg : block.getArguments()) {
              if (auto mrTy = dyn_cast<MemRefType>(arg.getType())) {
                auto newTy = convertMemRefType(mrTy, ctx);
                if (newTy != mrTy)
                  arg.setType(newTy);
              }
            }
          }
        }
      });
    }
  }
};

} // namespace

namespace mlir::aster::mlir_air {
std::unique_ptr<Pass> createConvertMemSpaceToAMDGCN() {
  return std::make_unique<ConvertMemSpaceToAMDGCN>();
}
} // namespace mlir::aster::mlir_air
