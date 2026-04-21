// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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

// Returns the enclosing transform.payload of `op`, if any. Wave passes operate
// on `mlir::ModuleOp` anchors, but the relevant normal form information is
// stored on the `transform::PayloadOp` that wraps the module.
static transform::PayloadOp getEnclosingPayload(Operation *op) {
  if (auto payload = llvm::dyn_cast<transform::PayloadOp>(op))
    return payload;
  return op->getParentOfType<transform::PayloadOp>();
}

bool wave::isInsideTransformPayload(Operation *op) {
  return static_cast<bool>(getEnclosingPayload(op));
}

// Returns the transform.payload ops to operate on for the given pass root.
// If `root` is itself enclosed in a payload, returns just that enclosing
// payload. Otherwise, returns every transform.payload op nested anywhere
// within `root`. This makes the wave passes work both when invoked directly
// on the inner module and when invoked on a top-level module that contains
// one or more transform.payload regions.
static SmallVector<transform::PayloadOp>
collectRelevantPayloads(Operation *root) {
  SmallVector<transform::PayloadOp> payloads;
  if (auto enclosing = getEnclosingPayload(root)) {
    payloads.push_back(enclosing);
    return payloads;
  }
  root->walk([&](transform::PayloadOp p) { payloads.push_back(p); });
  return payloads;
}

// Helper that returns the array of normal forms attached to a transform.payload
// op as a SmallVector of NormalFormAttrInterface.
static SmallVector<transform::NormalFormAttrInterface>
getPayloadNormalForms(transform::PayloadOp payload) {
  SmallVector<transform::NormalFormAttrInterface> result;
  llvm::append_range(result,
                     payload.getNormalForms()
                         .getAsRange<transform::NormalFormAttrInterface>());
  return result;
}

// Helper that updates the `normal_forms` attribute of a transform.payload op
// from the given set of attributes.
static void setPayloadNormalForms(
    transform::PayloadOp payload,
    ArrayRef<transform::NormalFormAttrInterface> normalForms) {
  SmallVector<Attribute> attrs(normalForms.begin(), normalForms.end());
  payload.setNormalFormsAttr(ArrayAttr::get(payload.getContext(), attrs));
}

// Sets the normal form on a single `payload` op, preserving the existing
// wave normal form bits when `preserve` is true.
static void setPayloadPostcondition(transform::PayloadOp payload,
                                    wave::WaveWaterNormalForm form,
                                    bool preserve) {
  SmallVector<transform::NormalFormAttrInterface> currentForms =
      getPayloadNormalForms(payload);

  wave::WaveWaterNormalForm finalForm = form;
  SmallVector<transform::NormalFormAttrInterface> newForms;
  newForms.reserve(currentForms.size() + 1);
  for (transform::NormalFormAttrInterface nf : currentForms) {
    auto waveForm = llvm::dyn_cast<wave::WaveWaterNormalFormAttr>(nf);
    if (!waveForm) {
      newForms.push_back(nf);
      continue;
    }
    if (preserve)
      finalForm = finalForm | waveForm.getValue();
  }
  newForms.push_back(
      wave::WaveWaterNormalFormAttr::get(payload.getContext(), finalForm));
  setPayloadNormalForms(payload, newForms);
}

llvm::LogicalResult
wave::setWaterNormalFormPassPostcondition(wave::WaveWaterNormalForm form,
                                          Operation *root, bool preserve) {
  SmallVector<transform::PayloadOp> payloads = collectRelevantPayloads(root);
  for (transform::PayloadOp payload : payloads)
    setPayloadPostcondition(payload, form, preserve);

  // We rely on the pass manager to call verifyRegion on the
  // transform.payload after the pass.
  return llvm::success();
}

llvm::LogicalResult
wave::clearWaterNormalFormPassPostcondition(Operation *root) {
  SmallVector<transform::PayloadOp> payloads = collectRelevantPayloads(root);
  for (transform::PayloadOp payload : payloads) {
    SmallVector<transform::NormalFormAttrInterface> currentForms =
        getPayloadNormalForms(payload);

    SmallVector<transform::NormalFormAttrInterface> remaining;
    remaining.reserve(currentForms.size());
    for (transform::NormalFormAttrInterface nf : currentForms) {
      if (!llvm::isa<WaveWaterNormalFormAttr>(nf))
        remaining.push_back(nf);
    }
    setPayloadNormalForms(payload, remaining);
  }
  return llvm::success();
}

llvm::LogicalResult wave::verifyWaterNormalFormPassPrecondition(
    WaveWaterNormalForm form, Operation *root, llvm::StringRef passName) {
  SmallVector<transform::PayloadOp> payloads = collectRelevantPayloads(root);
  for (transform::PayloadOp payload : payloads) {
    WaveWaterNormalForm expectedForm = WaveWaterNormalForm::None;
    for (Attribute attr : payload.getNormalForms()) {
      if (auto waveForm = llvm::dyn_cast<WaveWaterNormalFormAttr>(attr))
        expectedForm |= waveForm.getValue();
    }

    if (wave::bitEnumContainsAll(expectedForm, form))
      continue;

    return payload.emitError()
           << passName << " pass expects the payload to guarantee the "
           << wave::stringifyEnum(form) << " normal form";
  }
  return llvm::success();
}
