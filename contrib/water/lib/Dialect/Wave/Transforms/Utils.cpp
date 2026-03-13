// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/Utils.h"
#include "mlir/IR/Attributes.h"
#include "water/Dialect/NormalForm/IR/NormalFormInterfaces.h"
#include "water/Dialect/NormalForm/IR/NormalFormOps.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;

llvm::LogicalResult wave::collectWaveConstraints(
    Operation *top, llvm::DenseMap<Operation *, Attribute> &constraints) {
  auto *waveDialect = top->getContext()->getLoadedDialect<wave::WaveDialect>();
  auto walkResult = top->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto attr = op->getAttrOfType<ArrayAttr>(
            wave::WaveDialect::kWaveConstraintsAttrName)) {
      constraints[op] = attr;
      return WalkResult::skip();
    }
    if (op->getDialect() == waveDialect) {
      op->emitError()
          << "wave dialect operation without constraints on an ancestor";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return llvm::failure();
  return llvm::success();
}

llvm::LogicalResult
wave::setWaterNormalFormPassPostcondition(wave::WaveWaterNormalForm form,
                                          Operation *root, bool preserve) {
  auto module = llvm::dyn_cast<water_normalform::ModuleOp>(root);
  if (!module)
    return root->emitError() << "expected water_normalform.module";

  wave::WaveWaterNormalForm finalForm = form;
  auto water_normalforms =
      module.getNormalForms()
          .getAsRange<water_normalform::WaterNormalFormAttrInterface>();

  SmallVector<water_normalform::WaterNormalFormAttrInterface>
      waveWaterNormalForms = llvm::filter_to_vector(
          water_normalforms, llvm::IsaPred<wave::WaveWaterNormalFormAttr>);

  if (preserve) {
    // Merge all existing normal forms with the new form.
    for (auto nf : waveWaterNormalForms) {
      wave::WaveWaterNormalForm currentForm =
          cast<WaveWaterNormalFormAttr>(nf).getValue();
      finalForm = finalForm | currentForm;
    }
  }

  if (!waveWaterNormalForms.empty())
    module.removeWaterNormalForms(waveWaterNormalForms);

  module.addWaterNormalForms(
      {wave::WaveWaterNormalFormAttr::get(root->getContext(), finalForm)});

  // We rely on the pass manager to call verifyRegion on the
  // water_normalform.module after the pass

  return llvm::success();
}

llvm::LogicalResult
wave::clearWaterNormalFormPassPostcondition(Operation *root) {
  auto module = llvm::dyn_cast<water_normalform::ModuleOp>(root);
  if (!module)
    return root->emitError()
           << "expected << " << water_normalform::ModuleOp::getOperationName();

  auto water_normalforms =
      module.getNormalForms()
          .getAsRange<water_normalform::WaterNormalFormAttrInterface>();

  SmallVector<water_normalform::WaterNormalFormAttrInterface> toRemove =
      llvm::filter_to_vector(water_normalforms,
                             llvm::IsaPred<WaveWaterNormalFormAttr>);

  module.removeWaterNormalForms(toRemove);

  return llvm::success();
}

llvm::LogicalResult wave::verifyWaterNormalFormPassPrecondition(
    WaveWaterNormalForm form, Operation *root, llvm::StringRef passName) {
  auto module = llvm::dyn_cast<water_normalform::ModuleOp>(root);
  if (!module)
    return root->emitError()
           << "expected << " << water_normalform::ModuleOp::getOperationName();

  ArrayRef<Attribute> water_normalforms = module.getNormalForms().getValue();
  WaveWaterNormalForm expectedForm = WaveWaterNormalForm::None;
  for (Attribute form : llvm::make_filter_range(
           water_normalforms, llvm::IsaPred<WaveWaterNormalFormAttr>)) {
    expectedForm |= cast<WaveWaterNormalFormAttr>(form).getValue();
  }

  if (wave::bitEnumContainsAll(expectedForm, form))
    return llvm::success();

  return root->emitError()
         << passName
         << " pass expects the root operation or its ancestor to guarantee the "
         << wave::stringifyEnum(form) << " normal form";
}
