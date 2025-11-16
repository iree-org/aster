//===- ToAMDGCN.cpp -------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Transforms/Passes.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/ToAMDGCN.h"
#include "aster/Dialect/PIR/IR/PIRTypes.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Transforms/Utils.h"

namespace mlir {
namespace aster {
namespace amdgcn {
#define GEN_PASS_DEF_TOAMDGCN
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace aster
} // namespace mlir

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// ToAMDGCN pass
//===----------------------------------------------------------------------===//
struct ToAMDGCN : public amdgcn::impl::ToAMDGCNBase<ToAMDGCN> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ToAMDGCNConverter
//===----------------------------------------------------------------------===//

ToAMDGCNConverter::ToAMDGCNConverter(MLIRContext &context)
    : Builder(&context), context(&context) {
  addConversion([](Type type) -> Type { return type; });
  addConversion(
      [](pir::TypedRegisterType regTy) -> Type { return regTy.getReg(); });

  // Add generic source and target materializations.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
}

//===----------------------------------------------------------------------===//
// ToAMDGCNPatternBase
//===----------------------------------------------------------------------===//

Value ToAMDGCNPatternBase::createAllocation(RewriterBase &rewriter,
                                            Location loc, Type regTy) const {
  auto rTy = dyn_cast<RegisterTypeInterface>(converter.convertType(regTy));
  if (!rTy)
    rTy = dyn_cast<RegisterTypeInterface>(regTy);
  assert((isa<AGPRType, AGPRRangeType, SGPRType, SGPRRangeType, VGPRType,
              VGPRRangeType>(rTy)) &&
         "Expected a register type for allocation");
  if (!rTy.isRegisterRange())
    return amdgcn::AllocaOp::create(rewriter, loc, rTy);
  SmallVector<Value> results;
  RegisterRange range = rTy.getAsRange();
  results.reserve(range.size());
  for (int16_t i = 0; i < range.size(); ++i) {
    Value alloc = amdgcn::AllocaOp::create(
        rewriter, loc, rTy.cloneRegisterType(range.begin().getWithOffset(i)));
    results.push_back(alloc);
  }
  return MakeRegisterRangeOp::create(rewriter, loc, rTy, results);
}

ValueRange ToAMDGCNPatternBase::getOrSplitRange(RewriterBase &rewriter,
                                                Location loc,
                                                ValueRange values) const {
  if (values.size() != 1)
    return values;
  if (auto rTy = dyn_cast<RegisterTypeInterface>(values[0].getType());
      rTy && !rTy.isRegisterRange())
    return values;
  return SplitRegisterRangeOp::create(rewriter, loc, values[0]).getResults();
}

//===----------------------------------------------------------------------===//
// ToAMDGCN pass
//===----------------------------------------------------------------------===//

void ToAMDGCN::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  ToAMDGCNConverter converter(getContext());
  ConversionTarget target(getContext());
  populateFuncConversionPatterns(converter, target, patterns);
  populateToAMDGCNPatterns(converter, patterns, target);
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(
          op, target, FrozenRewritePatternSet(std::move(patterns)), config)))
    return signalPassFailure();
}
