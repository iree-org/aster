//===- TypeConverter.cpp - Type conversion utilities ----------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Legalizer.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::aster;

//===----------------------------------------------------------------------===//
// MemRefDescriptor
//===----------------------------------------------------------------------===//

FailureOr<MemRefDescriptor> MemRefDescriptor::get(MemRefType type,
                                                  ValueRange values) {
  if (values.empty())
    return failure();

  // Get static strides and offset.
  SmallVector<int64_t> staticStrides;
  int64_t staticOffset;
  if (failed(type.getStridesAndOffset(staticStrides, staticOffset)))
    return failure();

  int64_t numValues = 1; // aligned ptr
                         // Dynamic offset.
  numValues += ShapedType::isDynamic(staticOffset) ? 1 : 0;
  numValues += llvm::count_if(type.getShape(), ShapedType::isDynamic);
  numValues += llvm::count_if(staticStrides, ShapedType::isDynamic);

  // Bail out if the number of values does not match.
  if (static_cast<int64_t>(values.size()) != numValues || values.empty())
    return failure();

  MemRefDescriptor descriptor(type);
  descriptor.alignedPtr = values[0];

  MLIRContext *ctx = type.getContext();
  int64_t idx = 1; // Start after aligned ptr.

  // Helper to get OpFoldResult for static/dynamic value.
  auto getOpFoldResult = [&](int64_t staticValue) -> OpFoldResult {
    if (ShapedType::isDynamic(staticValue))
      return values[idx++];
    return IntegerAttr::get(IndexType::get(ctx), staticValue);
  };

  // Compute offset.
  descriptor.offset = getOpFoldResult(staticOffset);

  // Compute sizes.
  ArrayRef<int64_t> shape = type.getShape();
  descriptor.numSizes = shape.size();
  for (int64_t dim : shape)
    descriptor.sizesAndStrides.push_back(getOpFoldResult(dim));

  // Compute strides.
  for (int64_t stride : staticStrides)
    descriptor.sizesAndStrides.push_back(getOpFoldResult(stride));

  return descriptor;
}

LogicalResult MemRefDescriptor::convertType(const TypeConverter &converter,
                                            MemRefType type,
                                            SmallVectorImpl<Type> &results) {
  MLIRContext *ctx = type.getContext();

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return failure();

  // Convert the memory space attribute.
  ptr::MemorySpaceAttrInterface memSpace;
  if (Attribute ms = type.getMemorySpace()) {
    memSpace = dyn_cast_or_null<ptr::MemorySpaceAttrInterface>(
        converter.convertTypeAttribute(type, ms).value_or(nullptr));
    if (!memSpace)
      return failure();
  } else {
    memSpace = ptr::GenericSpaceAttr::get(ctx);
  }

  // 1. Aligned pointer.
  results.push_back(ptr::PtrType::get(ctx, memSpace));

  // 2. Dynamic offset (if present).
  if (ShapedType::isDynamic(offset))
    results.push_back(IndexType::get(ctx));

  // 3. One index for each dynamic size.
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim))
      results.push_back(IndexType::get(ctx));
  }

  // 4. One index for each dynamic stride.
  for (int64_t stride : strides) {
    if (ShapedType::isDynamic(stride))
      results.push_back(IndexType::get(ctx));
  }
  return success();
}

OpFoldResult MemRefDescriptor::getSize(int64_t i) const {
  assert(i >= 0 && i < getNumSizes() && "size index out of bounds");
  return sizesAndStrides[i];
}

OpFoldResult MemRefDescriptor::getStride(int64_t i) const {
  assert(i >= 0 && i < getNumStrides() && "stride index out of bounds");
  return sizesAndStrides[numSizes + i];
}

//===----------------------------------------------------------------------===//
// LegalizerTypeConverter
//===----------------------------------------------------------------------===//

LegalizerTypeConverter::LegalizerTypeConverter(MLIRContext *ctx) {
  // Default: keep types as-is.
  addConversion([](Type type) { return type; });

  // Convert memory space attributes.
  addTypeAttributeConversion(
      [](BaseMemRefType type,
         Attribute memorySpaceAttr) -> AttributeConversionResult {
        // Preserve ptr memory space attributes.
        if (auto memSpace =
                dyn_cast<ptr::MemorySpaceAttrInterface>(memorySpaceAttr)) {
          return memSpace;
        }

        // Convert gpu memory space attributes to amdgcn equivalents.
        if (auto space = dyn_cast<gpu::AddressSpaceAttr>(memorySpaceAttr)) {
          auto rwAccess = amdgcn::AccessKind::ReadWrite;
          if (space.getValue() == gpu::AddressSpace::Private) {
            return amdgcn::AddressSpaceAttr::get(
                type.getContext(), amdgcn::AddressSpaceKind::Private, rwAccess);
          }
          if (space.getValue() == gpu::AddressSpace::Global) {
            return amdgcn::AddressSpaceAttr::get(
                type.getContext(), amdgcn::AddressSpaceKind::Global, rwAccess);
          }
          if (space.getValue() == gpu::AddressSpace::Workgroup) {
            return amdgcn::AddressSpaceAttr::get(
                type.getContext(), amdgcn::AddressSpaceKind::Local, rwAccess);
          }
        }
        return AttributeConversionResult::abort();
      });

  // MemRef 1-to-N conversion.
  addConversion([&](MemRefType type, SmallVectorImpl<Type> &results) {
    return MemRefDescriptor::convertType(*this, type, results);
  });

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
