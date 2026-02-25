//===- MemOpsLegalization.cpp - Legalize memory operations ----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Legalizer.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrEnums.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
/// Flatten indices from ArrayRef<ValueRange> (1-to-N conversion result) to a
/// flat SmallVector<Value>.
static SmallVector<Value> flattenIndices(ArrayRef<ValueRange> indices) {
  SmallVector<Value> flat;
  for (auto &ins : indices)
    flat.append(ins.begin(), ins.end());
  return flat;
}

/// Compute the byte offset for accessing a memref element at the given indices.
/// Returns: (offset + sum(indices[i] * strides[i])) * sizeof(elementType)
static Value computeByteOffset(OpBuilder &builder, Location loc,
                               const aster::MemRefDescriptor &descriptor,
                               ValueRange indices, Type elementType) {
  int64_t rank = descriptor.getNumSizes();

  // Collect strides as OpFoldResults.
  SmallVector<OpFoldResult> strides;
  for (int64_t i = 0; i < rank; ++i)
    strides.push_back(descriptor.getStride(i));

  // Collect indices as OpFoldResults.
  SmallVector<OpFoldResult> indexOFRs = getAsOpFoldResult(indices);

  // Use upstream utility: offset + sum(indices[i] * strides[i]).
  auto [linearExpr, linearOperands] =
      computeLinearIndex(descriptor.getOffset(), strides, indexOFRs);
  OpFoldResult linearOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, linearExpr, linearOperands);

  // Multiply by element size in bytes.
  Type indexType = builder.getIndexType();
  Value typeOffset = builder.create<ptr::TypeOffsetOp>(
      loc, indexType, TypeAttr::get(elementType));

  // byteOffset = linearOffset * typeOffset
  AffineExpr d0, d1;
  bindDims(builder.getContext(), d0, d1);
  OpFoldResult byteOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, d0 * d1, {linearOffset, OpFoldResult(typeOffset)});
  return getValueOrCreateConstantIndexOp(builder, loc, byteOffset);
}

/// Compute a pointer to the memref element at the given indices.
/// Returns: basePtr + byteOffset
static Value computeElementPtr(OpBuilder &builder, Location loc,
                               const aster::MemRefDescriptor &descriptor,
                               ValueRange indices, Type elementType) {
  Value basePtr = descriptor.getAlignedPtr();
  auto ptrType = cast<ptr::PtrType>(basePtr.getType());

  Value byteOffset =
      computeByteOffset(builder, loc, descriptor, indices, elementType);

  // inbounds: memref load/store requires in-bounds indices; out-of-bounds is
  // UB.
  return builder.create<ptr::PtrAddOp>(loc, ptrType, basePtr, byteOffset,
                                       ptr::PtrAddFlags::inbounds);
}

/// Compute a pointer to the memref element at the given indices.
/// Converts the memref ValueRange to a MemRefDescriptor first.
static FailureOr<Value> computeElementPtr(OpBuilder &builder, Location loc,
                                          MemRefType memrefType,
                                          ValueRange memrefValues,
                                          ValueRange indices) {
  auto descriptorOrFailure =
      aster::MemRefDescriptor::get(memrefType, memrefValues);
  if (failed(descriptorOrFailure))
    return failure();

  return computeElementPtr(builder, loc, *descriptorOrFailure, indices,
                           memrefType.getElementType());
}

//===----------------------------------------------------------------------===//
// memref.dim -> constant or dynamic size (1-to-N conversion)
//===----------------------------------------------------------------------===//

/// Convert memref.dim with a constant index to the appropriate size value.
struct MemRefDimToValue : public OpConversionPattern<memref::DimOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the constant index.
    auto indexOp = op.getIndex().getDefiningOp<arith::ConstantIndexOp>();
    if (!indexOp)
      return rewriter.notifyMatchFailure(op, "dim index is not a constant");

    int64_t dimIndex = indexOp.value();
    MemRefType memrefType = cast<MemRefType>(op.getSource().getType());

    // Check bounds.
    if (dimIndex < 0 || dimIndex >= memrefType.getRank())
      return rewriter.notifyMatchFailure(op, "dim index out of bounds");

    // Get the descriptor from converted memref values.
    auto descriptorOrFailure =
        aster::MemRefDescriptor::get(memrefType, adaptor.getSource());
    if (failed(descriptorOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to get MemRefDescriptor from converted values");
    auto descriptor = *descriptorOrFailure;

    // Get the size at the given dimension.
    OpFoldResult size = descriptor.getSize(dimIndex);
    Value sizeVal = getValueOrCreateConstantIndexOp(rewriter, loc, size);

    rewriter.replaceOp(op, sizeVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// memref.extract_strided_metadata -> descriptor values (1-to-N conversion)
//===----------------------------------------------------------------------===//

/// Convert memref.extract_strided_metadata to the decomposed descriptor values.
struct ExtractStridedMetadataToValues
    : public OpConversionPattern<memref::ExtractStridedMetadataOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = cast<MemRefType>(op.getSource().getType());
    int64_t rank = memrefType.getRank();

    // Get the descriptor from converted memref values.
    auto descriptorOrFailure =
        aster::MemRefDescriptor::get(memrefType, adaptor.getSource());
    if (failed(descriptorOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to get MemRefDescriptor from converted values");
    auto descriptor = *descriptorOrFailure;

    // Build the replacement values for each result.
    // Results: baseBuffer, offset, sizes..., strides...
    SmallVector<SmallVector<Value>> replacements;

    // 1. Base buffer - this becomes the aligned pointer.
    replacements.push_back({descriptor.getAlignedPtr()});

    // 2. Offset.
    Value offsetVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, descriptor.getOffset());
    replacements.push_back({offsetVal});

    // 3. Sizes.
    for (int64_t i = 0; i < rank; ++i) {
      Value sizeVal =
          getValueOrCreateConstantIndexOp(rewriter, loc, descriptor.getSize(i));
      replacements.push_back({sizeVal});
    }

    // 4. Strides.
    for (int64_t i = 0; i < rank; ++i) {
      Value strideVal = getValueOrCreateConstantIndexOp(
          rewriter, loc, descriptor.getStride(i));
      replacements.push_back({strideVal});
    }

    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// memref.load -> ptr.load pattern (1-to-N conversion)
//===----------------------------------------------------------------------===//

/// Convert memref.load to ptr.load using the decomposed memref descriptor.
struct MemRefLoadToPtr : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = op.getMemRefType();

    // Compute element pointer.
    auto ptrOrFailure =
        computeElementPtr(rewriter, loc, memrefType, adaptor.getMemref(),
                          flattenIndices(adaptor.getIndices()));
    if (failed(ptrOrFailure))
      return rewriter.notifyMatchFailure(op,
                                         "failed to compute element pointer");

    // Load the value.
    rewriter.replaceOpWithNewOp<ptr::LoadOp>(op, op.getResult().getType(),
                                             *ptrOrFailure);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// memref.store -> ptr.store pattern (1-to-N conversion)
//===----------------------------------------------------------------------===//

/// Convert memref.store to ptr.store using the decomposed memref descriptor.
struct MemRefStoreToPtr : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = op.getMemRefType();

    // Compute element pointer.
    auto ptrOrFailure =
        computeElementPtr(rewriter, loc, memrefType, adaptor.getMemref(),
                          flattenIndices(adaptor.getIndices()));
    if (failed(ptrOrFailure))
      return rewriter.notifyMatchFailure(op,
                                         "failed to compute element pointer");

    // Get the value to store (should be 1-to-1 for scalar types).
    ValueRange valueToStore = adaptor.getValue();
    assert(valueToStore.size() == 1 && "expected single value to store");

    // Store the value.
    rewriter.replaceOpWithNewOp<ptr::StoreOp>(op, valueToStore.front(),
                                              *ptrOrFailure);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// vector.load -> ptr.load pattern (1-to-N conversion)
//===----------------------------------------------------------------------===//

/// Convert vector.load to ptr.load using the decomposed memref descriptor.
struct VectorLoadToPtr : public OpConversionPattern<vector::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::LoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = cast<MemRefType>(op.getBase().getType());

    // Compute element pointer.
    auto ptrOrFailure =
        computeElementPtr(rewriter, loc, memrefType, adaptor.getBase(),
                          flattenIndices(adaptor.getIndices()));
    if (failed(ptrOrFailure))
      return rewriter.notifyMatchFailure(op,
                                         "failed to compute element pointer");

    // Load the vector.
    rewriter.replaceOpWithNewOp<ptr::LoadOp>(op, op.getResult().getType(),
                                             *ptrOrFailure);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// vector.store -> ptr.store pattern (1-to-N conversion)
//===----------------------------------------------------------------------===//

/// Convert vector.store to ptr.store using the decomposed memref descriptor.
struct VectorStoreToPtr : public OpConversionPattern<vector::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::StoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemRefType memrefType = cast<MemRefType>(op.getBase().getType());

    // Compute element pointer.
    auto ptrOrFailure =
        computeElementPtr(rewriter, loc, memrefType, adaptor.getBase(),
                          flattenIndices(adaptor.getIndices()));
    if (failed(ptrOrFailure))
      return rewriter.notifyMatchFailure(op,
                                         "failed to compute element pointer");

    // Get the value to store (should be 1-to-1 for vector types).
    ValueRange valueToStore = adaptor.getValueToStore();
    assert(valueToStore.size() == 1 && "expected single value to store");

    // Store the vector.
    rewriter.replaceOpWithNewOp<ptr::StoreOp>(op, valueToStore.front(),
                                              *ptrOrFailure);
    return success();
  }
};

} // namespace

void mlir::aster::populateMemOpsConversionPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns
      .add<MemRefDimToValue, ExtractStridedMetadataToValues, MemRefLoadToPtr,
           MemRefStoreToPtr, VectorLoadToPtr, VectorStoreToPtr>(
          converter, patterns.getContext(), benefit);
}
