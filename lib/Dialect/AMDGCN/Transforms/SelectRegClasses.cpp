//===- SelectRegClasses.cpp -----------------------------------------------===//
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
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Transforms/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_SELECTREGCLASSES
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// SelectRegClasses pass
//===----------------------------------------------------------------------===//
struct SelectRegClasses
    : public amdgcn::impl::SelectRegClassesBase<SelectRegClasses> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// TypeConverter
//===----------------------------------------------------------------------===//
struct ColorConverter : TypeConverter {
  ColorConverter();
};
} // namespace

//===----------------------------------------------------------------------===//
// TypeConverter
//===----------------------------------------------------------------------===//

ColorConverter::ColorConverter() {
  addConversion([&](Type type) { return type; });
  addConversion([&](GGPRType type) -> Type {
    RegisterRange range = type.getAsRange();
    if (std::optional<bool> isUniform = type.getIsUniform();
        isUniform && *isUniform) {
      if (range.size() == 1)
        return SGPRType::get(type.getContext(), Register());
      return SGPRRangeType::get(type.getContext(), range);
    }
    if (range.size() == 1)
      return VGPRType::get(type.getContext(), Register());
    return VGPRRangeType::get(type.getContext(), range);
  });
}

//===----------------------------------------------------------------------===//
// SelectRegClasses pass
//===----------------------------------------------------------------------===//

// TODO: Don't do this.
template <typename Op, typename... Ops>
void addPattens(RewritePatternSet &patterns, ColorConverter &converter) {
  patterns.add<GenericOpConversion<Op>>(converter, patterns.getContext());
  if constexpr (sizeof...(Ops) > 0) {
    addPattens<Ops...>(patterns, converter);
  }
}

void SelectRegClasses::runOnOperation() {
  Operation *op = getOperation();
  ColorConverter converter;
  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  target.addDynamicallyLegalOp<
#define GET_OP_LIST
#include "aster/Dialect/LSIR/IR/LSIROps.cpp.inc"
      >([&](Operation *op) { return converter.isLegal(op); });
  addPattens<
#define GET_OP_LIST
#include "aster/Dialect/LSIR/IR/LSIROps.cpp.inc"
      >(patterns, converter);
  patterns.add<GenericOpConversion<TestInstOp>>(converter,
                                                patterns.getContext());
  populateFuncConversionPatterns(converter, target, patterns);
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(
          op, target, FrozenRewritePatternSet(std::move(patterns)), config)))
    return signalPassFailure();
}
