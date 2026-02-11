//===- DestructureStructIterArgs.cpp - Destructure struct iter_args -------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces struct-typed iter_args in scf.for loops with individual
// field-typed iter_args. This is needed after SCF pipelining, which may create
// struct-typed iter_args when cross-stage values are bundled in structs.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_DESTRUCTURESTRUCTITERARGS
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {

/// Check if an scf.for has any struct-typed iter_args.
static bool hasStructIterArgs(scf::ForOp forOp) {
  for (auto arg : forOp.getInitArgs()) {
    if (isa<StructType>(arg.getType()))
      return true;
  }
  return false;
}

/// Build the expanded init args list, inserting struct_extract ops as needed.
/// Returns the mapping from original iter_arg index to the starting index
/// in the expanded list, and the number of expanded args per original arg.
static SmallVector<Value>
buildExpandedInitArgs(OpBuilder &builder, Location loc, OperandRange initArgs,
                      SmallVector<std::pair<unsigned, unsigned>> &argMapping) {
  SmallVector<Value> expandedInits;
  for (auto [i, init] : llvm::enumerate(initArgs)) {
    auto structType = dyn_cast<StructType>(init.getType());
    if (!structType) {
      // Non-struct: pass through as-is.
      argMapping.push_back({(unsigned)expandedInits.size(), 1});
      expandedInits.push_back(init);
      continue;
    }
    // Struct: extract each field.
    unsigned startIdx = expandedInits.size();
    unsigned numFields = structType.getNumFields();
    SmallVector<Attribute> nameAttrs;
    SmallVector<Type> fieldTypes;
    for (size_t f = 0; f < numFields; ++f) {
      nameAttrs.push_back(structType.getFieldName(f));
      fieldTypes.push_back(structType.getFieldType(f));
    }
    auto extractOp = StructExtractOp::create(builder, loc, fieldTypes, init,
                                             builder.getArrayAttr(nameAttrs));
    for (auto result : extractOp.getResults())
      expandedInits.push_back(result);
    argMapping.push_back({startIdx, numFields});
  }
  return expandedInits;
}

/// Destructure a single scf.for op by replacing struct-typed iter_args
/// with individual field-typed iter_args.
/// Returns true if the operation was modified.
static bool destructureForOp(scf::ForOp forOp) {
  if (!hasStructIterArgs(forOp))
    return false;

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  // Step 1: Build expanded init args with field-level granularity.
  // argMapping[i] = (startIndex, count) in the expanded list.
  SmallVector<std::pair<unsigned, unsigned>> argMapping;
  SmallVector<Value> expandedInits =
      buildExpandedInitArgs(builder, loc, forOp.getInitArgs(), argMapping);

  // Step 2: Build expanded result types.
  SmallVector<Type> expandedResultTypes;
  for (auto init : expandedInits)
    expandedResultTypes.push_back(init.getType());

  // Step 3: Create the new scf.for with expanded iter_args.
  auto newForOp =
      scf::ForOp::create(builder, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), expandedInits);

  // Step 4: Set up the new loop body. The new for has a fresh body with
  // block args: [iv, expanded_iter_args...]. We need to reconstruct structs
  // from individual args so the original body can use them.
  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();

  // 4.a: Remove the auto-generated yield if present.
  if (newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  // 4.b: Build struct_create ops at the top of the new body to reconstruct
  // struct values from individual block args.
  builder.setInsertionPointToStart(newBody);

  // 4.c: Map old IV to new IV.
  IRMapping bodyMapping;
  bodyMapping.map(oldBody->getArgument(0), newBody->getArgument(0));

  // 4.d: Map old iter_arg block args to reconstructed structs (or direct args).
  unsigned numOrigIterArgs = forOp.getInitArgs().size();
  for (unsigned i = 0; i < numOrigIterArgs; ++i) {
    auto [startIdx, count] = argMapping[i];
    // +1 for the IV block arg.
    Value oldArg = oldBody->getArgument(i + 1);

    if (count == 1 && !isa<StructType>(oldArg.getType())) {
      // Non-struct: direct mapping.
      bodyMapping.map(oldArg, newBody->getArgument(startIdx + 1));
      continue;
    }

    // Struct: reconstruct from individual block args.
    SmallVector<Value> fields;
    for (unsigned f = 0; f < count; ++f)
      fields.push_back(newBody->getArgument(startIdx + 1 + f));

    auto structType = cast<StructType>(oldArg.getType());
    auto createOp = StructCreateOp::create(builder, loc, structType, fields);
    bodyMapping.map(oldArg, createOp.getResult());
  }

  // Step 5: Clone all ops from old body into new body (except the yield).
  auto yieldOp = cast<scf::YieldOp>(oldBody->getTerminator());
  for (auto &op : oldBody->without_terminator()) {
    builder.clone(op, bodyMapping);
  }

  // Step 6: Build the new yield with expanded values.
  SmallVector<Value> expandedYieldOperands;
  for (unsigned i = 0; i < numOrigIterArgs; ++i) {
    Value oldYieldVal = bodyMapping.lookupOrDefault(yieldOp.getOperand(i));
    auto structType = dyn_cast<StructType>(forOp.getInitArgs()[i].getType());
    if (!structType) {
      expandedYieldOperands.push_back(oldYieldVal);
      continue;
    }
    // 6.a: Struct: extract fields for yield.
    unsigned numFields = structType.getNumFields();
    SmallVector<Attribute> nameAttrs;
    SmallVector<Type> fieldTypes;
    for (size_t f = 0; f < numFields; ++f) {
      nameAttrs.push_back(structType.getFieldName(f));
      fieldTypes.push_back(structType.getFieldType(f));
    }
    auto extractOp = StructExtractOp::create(
        builder, loc, fieldTypes, oldYieldVal, builder.getArrayAttr(nameAttrs));
    for (auto result : extractOp.getResults())
      expandedYieldOperands.push_back(result);
  }
  scf::YieldOp::create(builder, loc, expandedYieldOperands);

  // Step 7: Replace uses of old for results with reconstructed structs.
  builder.setInsertionPointAfter(newForOp);
  for (unsigned i = 0; i < numOrigIterArgs; ++i) {
    Value oldResult = forOp.getResult(i);
    auto [startIdx, count] = argMapping[i];

    if (count == 1 && !isa<StructType>(oldResult.getType())) {
      oldResult.replaceAllUsesWith(newForOp.getResult(startIdx));
      continue;
    }

    // 7.a: Struct: reconstruct from expanded results.
    SmallVector<Value> fields;
    for (unsigned f = 0; f < count; ++f)
      fields.push_back(newForOp.getResult(startIdx + f));

    auto structType = cast<StructType>(oldResult.getType());
    auto createOp = StructCreateOp::create(builder, loc, structType, fields);
    oldResult.replaceAllUsesWith(createOp.getResult());
  }

  // Step 8: Erase the old for op.
  forOp.erase();
  return true;
}

//===----------------------------------------------------------------------===//
// DestructureStructIterArgs pass
//===----------------------------------------------------------------------===//

struct DestructureStructIterArgs
    : public aster_utils::impl::DestructureStructIterArgsBase<
          DestructureStructIterArgs> {
public:
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void DestructureStructIterArgs::runOnOperation() {
  bool changed = true;
  while (changed) {
    changed = false;
    // Collect all scf.for ops with struct iter_args. Process in reverse order
    // so inner loops are destructured before outer loops.
    SmallVector<scf::ForOp> worklist;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (hasStructIterArgs(forOp))
        worklist.push_back(forOp);
    });
    for (auto forOp : llvm::reverse(worklist)) {
      changed |= destructureForOp(forOp);
    }
  }
}
