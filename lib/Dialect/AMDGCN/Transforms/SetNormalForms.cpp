//===- SetNormalForms.cpp - Set normal form attributes on container ops
//----===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.h"
#include "mlir/AsmParser/AsmParser.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_SETNORMALFORMS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
struct SetNormalForms
    : public amdgcn::impl::SetNormalFormsBase<SetNormalForms> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Parse a list of mnemonic strings into NormalFormAttrInterface attributes.
  /// Returns failure if any mnemonic is invalid.
  FailureOr<SmallVector<normalform::NormalFormAttrInterface>>
  parseFormMnemonics(ArrayRef<std::string> mnemonics);
};
} // namespace

FailureOr<SmallVector<normalform::NormalFormAttrInterface>>
SetNormalForms::parseFormMnemonics(ArrayRef<std::string> mnemonics) {
  MLIRContext *ctx = &getContext();
  SmallVector<normalform::NormalFormAttrInterface> attrs;
  for (const std::string &mnemonic : mnemonics) {
    std::string attrStr = "#amdgcn." + mnemonic;
    Attribute attr = mlir::parseAttribute(attrStr, ctx);
    if (!attr) {
      getOperation()->emitError()
          << "unknown normal form mnemonic: '" << mnemonic << "' "
          << "(tried to parse as '" << attrStr << "')";
      return failure();
    }
    auto nfAttr = dyn_cast<normalform::NormalFormAttrInterface>(attr);
    if (!nfAttr) {
      getOperation()->emitError()
          << "attribute '" << attrStr
          << "' does not implement NormalFormAttrInterface";
      return failure();
    }
    attrs.push_back(nfAttr);
  }
  return attrs;
}

void SetNormalForms::runOnOperation() {
  Operation *op = getOperation();

  auto moduleAttrs = parseFormMnemonics(moduleForms);
  if (failed(moduleAttrs))
    return signalPassFailure();

  auto kernelAttrs = parseFormMnemonics(kernelForms);
  if (failed(kernelAttrs))
    return signalPassFailure();

  if (!moduleAttrs->empty()) {
    op->walk([&](amdgcn::ModuleOp moduleOp) {
      moduleOp.addNormalForms(*moduleAttrs);
    });
  }

  if (!kernelAttrs->empty()) {
    op->walk([&](KernelOp kernelOp) { kernelOp.addNormalForms(*kernelAttrs); });
  }
}
