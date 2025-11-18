//===- UpstreamExternalModels.cpp -------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/GPUFuncInterface.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "aster/Interfaces/UpstreamExternalModels.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::aster;

namespace {
/// External model implementation for func::FuncOp
struct FuncOpGPUFuncInterfaceImpl
    : public GPUFuncInterface::ExternalModel<FuncOpGPUFuncInterfaceImpl,
                                             func::FuncOp> {

  /// Attribute names for storing GPU metadata
  static constexpr StringRef kGridDimsAttr = "gpu.grid_dims";
  static constexpr StringRef kBlockDimsAttr = "gpu.block_dims";
  static constexpr StringRef kGPUKernelAttr = "gpu.kernel";
  static constexpr StringRef kHostABIAttr = "gpu.host_abi";
  static constexpr StringRef kSharedMemorySizeAttr = "gpu.shared_memory_size";

  /// Sets a dimension attribute on the given operation.
  static void setDimAttr(Operation *op, ArrayRef<int32_t> dims,
                         StringRef name) {
    if (dims.empty()) {
      op->removeAttr(name);
      return;
    }
    assert(dims.size() <= 3 && "Expected at most 3 dimensions");
    Builder builder(op->getContext());
    SmallVector<int32_t, 3> dimValues;
    llvm::append_range(dimValues, dims);
    while (dimValues.size() < 3)
      dimValues.push_back(1);
    op->setAttr(name, builder.getDenseI32ArrayAttr(dimValues));
  }
  ArrayRef<int32_t> getGridDims(Operation *op) const {
    if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(kGridDimsAttr))
      return attr.asArrayRef();
    return {};
  }
  ArrayRef<int32_t> getBlockDims(Operation *op) const {
    if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(kBlockDimsAttr))
      return attr.asArrayRef();
    return {};
  }
  void setGridDims(Operation *op, ArrayRef<int32_t> dims) const {
    setDimAttr(op, dims, kGridDimsAttr);
  }
  void setBlockDims(Operation *op, ArrayRef<int32_t> dims) const {
    setDimAttr(op, dims, kBlockDimsAttr);
  }
  bool isGPUKernel(Operation *op) const {
    return op->getAttrOfType<UnitAttr>(kGPUKernelAttr) != nullptr;
  }
  void setGPUKernel(Operation *op, bool isKernel) const {
    if (!isKernel) {
      op->removeAttr(kGPUKernelAttr);
      return;
    }
    Builder builder(op->getContext());
    op->setAttr(kGPUKernelAttr, builder.getUnitAttr());
  }
  std::tuple<FunctionType, ArrayRef<int32_t>, ArrayRef<int32_t>>
  getHostABI(Operation *op) const {
    auto dictAttr = op->getAttrOfType<DictionaryAttr>(kHostABIAttr);
    if (!dictAttr)
      return {nullptr, {}, {}};

    FunctionType funcType = nullptr;
    if (auto typeAttr = dictAttr.getAs<TypeAttr>("type"))
      funcType = dyn_cast<FunctionType>(typeAttr.getValue());

    ArrayRef<int32_t> sizes;
    if (auto sizeAttr = dictAttr.getAs<DenseI32ArrayAttr>("size"))
      sizes = sizeAttr.asArrayRef();

    ArrayRef<int32_t> aligns;
    if (auto alignAttr = dictAttr.getAs<DenseI32ArrayAttr>("align"))
      aligns = alignAttr.asArrayRef();

    return {funcType, sizes, aligns};
  }
  void setHostABI(Operation *op, FunctionType type,
                  ArrayRef<int32_t> sizeInBytes,
                  ArrayRef<int32_t> alignInBytes) const {
    if (!type) {
      op->removeAttr(kHostABIAttr);
      return;
    }
    Builder builder(op->getContext());
    SmallVector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("type", TypeAttr::get(type)));
    if (!sizeInBytes.empty())
      attrs.push_back(builder.getNamedAttr(
          "size", builder.getDenseI32ArrayAttr(sizeInBytes)));
    if (!alignInBytes.empty())
      attrs.push_back(builder.getNamedAttr(
          "align", builder.getDenseI32ArrayAttr(alignInBytes)));
    op->setAttr(kHostABIAttr, builder.getDictionaryAttr(attrs));
  }
  int32_t getSharedMemorySize(Operation *op) const {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kSharedMemorySizeAttr))
      return attr.getInt();
    return 0;
  }
  void setSharedMemorySize(Operation *op, int32_t sizeInBytes) const {
    if (sizeInBytes <= 0) {
      op->removeAttr(kSharedMemorySizeAttr);
      return;
    }
    Builder builder(op->getContext());
    op->setAttr(kSharedMemorySizeAttr, builder.getI32IntegerAttr(sizeInBytes));
  }
};

//===----------------------------------------------------------------------===//
// ModuleOpInterface implementation for mlir::ModuleOp
//===----------------------------------------------------------------------===//

/// External model implementation for mlir::ModuleOp
struct ModuleOpModuleOpInterfaceImpl
    : public ModuleOpInterface::ExternalModel<ModuleOpModuleOpInterfaceImpl,
                                              mlir::ModuleOp> {
  // No methods to implement - this is a marker interface.
};
} // namespace

void mlir::aster::registerUpstreamExternalModels(DialectRegistry &registry) {
  registry.insert<func::FuncDialect>();
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    func::FuncOp::attachInterface<FuncOpGPUFuncInterfaceImpl>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    mlir::ModuleOp::attachInterface<ModuleOpModuleOpInterfaceImpl>(*ctx);
  });
}
