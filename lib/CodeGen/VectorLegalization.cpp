//===- VectorLegalization.cpp - Legalize vector operations ----------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Legalizer.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;

namespace {
//===----------------------------------------------------------------------===//
// ExtractOp -> ToElementsOp pattern
//===----------------------------------------------------------------------===//

/// Convert vector.extract with constant indices to vector.to_elements.
struct ExtractOpToElements : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasDynamicPosition())
      return rewriter.notifyMatchFailure(op, "dynamic positions not supported");

    auto sourceType = op.getSourceVectorType();
    ArrayRef<int64_t> staticPos = op.getStaticPosition();

    if (llvm::any_of(staticPos, [](int64_t idx) { return idx < 0; }))
      return rewriter.notifyMatchFailure(op, "poison indices");

    // Decompose the source vector into elements.
    auto toElements =
        rewriter.create<vector::ToElementsOp>(op.getLoc(), op.getSource());

    Type resultType = op.getResult().getType();

    // Handle scalar extraction.
    if (resultType.isIntOrIndexOrFloat()) {
      int64_t linearIdx =
          linearize(staticPos, computeSuffixProduct(sourceType.getShape()));
      rewriter.replaceOp(op, toElements.getResult(linearIdx));
      return success();
    }

    // Handle vector extraction.
    auto resultVecType = dyn_cast<VectorType>(resultType);
    if (!resultVecType)
      return rewriter.notifyMatchFailure(op, "unsupported result type");

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    ArrayRef<int64_t> resultShape = resultVecType.getShape();
    int64_t numResultElements = resultVecType.getNumElements();

    // Collect elements for the result vector.
    SmallVector<Value> resultElements;
    resultElements.reserve(numResultElements);

    for (int64_t i = 0; i < numResultElements; ++i) {
      // Delinearize to result indices.
      SmallVector<int64_t> resultIndices =
          delinearize(i, computeSuffixProduct(resultShape));

      // Build source indices: staticPos followed by resultIndices.
      SmallVector<int64_t> sourceIndices(staticPos.begin(), staticPos.end());
      sourceIndices.append(resultIndices.begin(), resultIndices.end());

      int64_t srcLinearIdx =
          linearize(sourceIndices, computeSuffixProduct(sourceShape));
      resultElements.push_back(toElements.getResult(srcLinearIdx));
    }

    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, resultVecType,
                                                        resultElements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InsertOp -> ToElementsOp + FromElementsOp pattern
//===----------------------------------------------------------------------===//

/// Convert vector.insert with constant indices to to_elements + from_elements.
struct InsertOpToFromElements : public OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasDynamicPosition())
      return rewriter.notifyMatchFailure(op, "dynamic positions not supported");

    auto destType = op.getDestVectorType();
    ArrayRef<int64_t> staticPos = op.getStaticPosition();

    if (llvm::any_of(staticPos, [](int64_t idx) { return idx < 0; }))
      return rewriter.notifyMatchFailure(op, "poison indices");

    // Decompose the destination vector into elements.
    auto destElements =
        rewriter.create<vector::ToElementsOp>(op.getLoc(), op.getDest());

    // Build the list of elements from destination.
    SmallVector<Value> elements(destElements.getElements().begin(),
                                destElements.getElements().end());

    Type valueType = op.getValueToStore().getType();
    ArrayRef<int64_t> destShape = destType.getShape();

    // Handle scalar insertion.
    if (valueType.isIntOrIndexOrFloat()) {
      int64_t linearIdx =
          linearize(staticPos, computeSuffixProduct(destShape));
      elements[linearIdx] = op.getValueToStore();
      rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, destType,
                                                          elements);
      return success();
    }

    // Handle vector insertion.
    auto valueVecType = dyn_cast<VectorType>(valueType);
    if (!valueVecType)
      return rewriter.notifyMatchFailure(op, "unsupported value type");

    // Decompose the value vector into elements.
    auto valueElements = rewriter.create<vector::ToElementsOp>(
        op.getLoc(), op.getValueToStore());

    ArrayRef<int64_t> valueShape = valueVecType.getShape();
    int64_t numValueElements = valueVecType.getNumElements();

    // Insert each element from the value vector into the destination.
    for (int64_t i = 0; i < numValueElements; ++i) {
      // Delinearize to value indices.
      SmallVector<int64_t> valueIndices =
          delinearize(i, computeSuffixProduct(valueShape));

      // Build dest indices: staticPos followed by valueIndices.
      SmallVector<int64_t> destIndices(staticPos.begin(), staticPos.end());
      destIndices.append(valueIndices.begin(), valueIndices.end());

      int64_t destLinearIdx =
          linearize(destIndices, computeSuffixProduct(destShape));
      elements[destLinearIdx] = valueElements.getResult(i);
    }

    // Reconstruct the vector.
    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, destType, elements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BroadcastOp -> FromElementsOp pattern
//===----------------------------------------------------------------------===//

/// Convert vector.broadcast from scalar to vector using from_elements.
struct BroadcastOpToFromElements
    : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    Type sourceType = source.getType();

    if (!sourceType.isIntOrIndexOrFloat())
      return rewriter.notifyMatchFailure(op,
                                         "only scalar-to-vector broadcasts");

    auto resultType = op.getResultVectorType();
    int64_t numElements = resultType.getNumElements();

    // Create a vector with all elements set to the scalar source.
    SmallVector<Value> elements(numElements, source);
    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, resultType,
                                                        elements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ExtractStridedSliceOp -> ToElementsOp + FromElementsOp pattern
//===----------------------------------------------------------------------===//

/// Convert vector.extract_strided_slice with constant offsets to elements ops.
struct ExtractStridedSliceOpToElements
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasNonUnitStrides())
      return rewriter.notifyMatchFailure(op, "non-unit strides not supported");

    auto sourceType = op.getSourceVectorType();
    auto resultType = cast<VectorType>(op.getResult().getType());

    // Get offsets and sizes.
    SmallVector<int64_t> offsets;
    op.getOffsets(offsets);

    // Decompose the source vector into elements.
    auto toElements =
        rewriter.create<vector::ToElementsOp>(op.getLoc(), op.getSource());

    // Collect elements for the result.
    int64_t numResultElements = resultType.getNumElements();
    SmallVector<Value> resultElements;
    resultElements.reserve(numResultElements);

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    // Iterate over all result elements.
    for (int64_t i = 0; i < numResultElements; ++i) {
      // Delinearize to result indices.
      SmallVector<int64_t> resultIndices =
          delinearize(i, computeSuffixProduct(resultShape));

      // Map result indices to source indices by adding offsets.
      SmallVector<int64_t> sourceIndices(sourceShape.size(), 0);
      for (size_t d = 0; d < offsets.size(); ++d) {
        sourceIndices[d] = resultIndices[d] + offsets[d];
      }
      // Trailing dimensions are copied directly.
      for (size_t d = offsets.size(); d < resultShape.size(); ++d) {
        sourceIndices[d] = resultIndices[d];
      }

      int64_t srcLinearIdx =
          linearize(sourceIndices, computeSuffixProduct(sourceShape));
      resultElements.push_back(toElements.getResult(srcLinearIdx));
    }

    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, resultType,
                                                        resultElements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InsertStridedSliceOp -> ToElementsOp + FromElementsOp pattern
//===----------------------------------------------------------------------===//

/// Convert vector.insert_strided_slice with constant offsets to elements ops.
struct InsertStridedSliceOpToElements
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasNonUnitStrides())
      return rewriter.notifyMatchFailure(op, "non-unit strides not supported");

    auto sourceType = op.getSourceVectorType();
    auto destType = op.getDestVectorType();

    // Get offsets.
    SmallVector<int64_t> offsets;
    for (auto attr : op.getOffsets())
      offsets.push_back(cast<IntegerAttr>(attr).getInt());

    // Decompose both vectors into elements.
    auto destElements =
        rewriter.create<vector::ToElementsOp>(op.getLoc(), op.getDest());
    auto srcElements = rewriter.create<vector::ToElementsOp>(
        op.getLoc(), op.getValueToStore());

    // Start with destination elements.
    SmallVector<Value> resultElements(destElements.getElements().begin(),
                                      destElements.getElements().end());

    ArrayRef<int64_t> destShape = destType.getShape();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();

    // Iterate over source elements and insert into destination.
    int64_t numSrcElements = sourceType.getNumElements();
    for (int64_t i = 0; i < numSrcElements; ++i) {
      // Delinearize to source indices.
      SmallVector<int64_t> srcIndices =
          delinearize(i, computeSuffixProduct(sourceShape));

      // Map source indices to destination indices by adding offsets.
      // Source is k-D, dest is n-D (n >= k), offsets is n-sized.
      // Source dimensions map to last k dimensions of dest.
      SmallVector<int64_t> destIndices(destShape.size(), 0);
      size_t srcRank = sourceShape.size();
      size_t destRank = destShape.size();

      // First (destRank - srcRank) dimensions use only offsets.
      for (size_t d = 0; d < destRank - srcRank; ++d) {
        destIndices[d] = offsets[d];
      }
      // Last srcRank dimensions add source indices to offsets.
      for (size_t d = 0; d < srcRank; ++d) {
        destIndices[destRank - srcRank + d] =
            offsets[destRank - srcRank + d] + srcIndices[d];
      }

      int64_t destLinearIdx =
          linearize(destIndices, computeSuffixProduct(destShape));
      resultElements[destLinearIdx] = srcElements.getResult(i);
    }

    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, destType,
                                                        resultElements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FromElementsOp leading unit dimensions pattern
//===----------------------------------------------------------------------===//

/// Returns the type with leading unit dims dropped, or nullopt if none exist.
/// If all dims are 1, returns a rank-0 vector type.
static std::optional<VectorType> dropLeadingUnitDims(VectorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  size_t n = 0;
  for (int64_t d : shape) {
    if (d != 1)
      break;
    ++n;
  }
  if (n == 0)
    return std::nullopt;
  return VectorType::get(
      SmallVector<int64_t>(shape.begin() + n, shape.end()),
      type.getElementType(), type.getScalableDims().drop_front(n));
}

/// Drop leading unit dimensions from vector.from_elements, then shape_cast
/// back.
struct FromElementsDropLeadingUnitDims
    : public OpRewritePattern<vector::FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    auto newVecType = dropLeadingUnitDims(resultType);
    if (!newVecType)
      return rewriter.notifyMatchFailure(op, "no leading unit dims to drop");

    // Create from_elements with the reduced shape.
    Value reducedVec = rewriter.create<vector::FromElementsOp>(
        op.getLoc(), *newVecType, op.getElements());

    // Shape cast back to the original type.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resultType,
                                                     reducedVec);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// StoreOp leading unit dimensions pattern
//===----------------------------------------------------------------------===//

/// Cast away leading unit dimensions from vector.store values.
struct StoreOpDropLeadingUnitDims : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value valueToStore = op.getValueToStore();
    auto vecType = dyn_cast<VectorType>(valueToStore.getType());
    if (!vecType)
      return rewriter.notifyMatchFailure(op, "stored value is not a vector");

    auto newVecType = dropLeadingUnitDims(vecType);
    if (!newVecType)
      return rewriter.notifyMatchFailure(op, "no leading unit dims to drop");

    // Create shape cast to remove leading unit dimensions.
    Value castValue = rewriter.create<vector::ShapeCastOp>(
        op.getLoc(), *newVecType, valueToStore);

    // Replace the store with one using the cast value.
    rewriter.replaceOpWithNewOp<vector::StoreOp>(op, castValue, op.getBase(),
                                                 op.getIndices());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// From/ToElementsOp rank-0 patterns
//===----------------------------------------------------------------------===//

/// Convert vector.from_elements with rank-0 result to return scalar directly.
struct FromElementsRank0Pattern
    : public OpConversionPattern<vector::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::FromElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    if (resultType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "not a rank-0 vector");

    // For rank-0 vector, just return the single scalar element.
    assert(adaptor.getElements().size() == 1 &&
           "rank-0 vector should have exactly one element");
    rewriter.replaceOp(op, adaptor.getElements()[0]);
    return success();
  }
};

/// Convert vector.to_elements with rank-0 input to return scalar directly.
struct ToElementsRank0Pattern
    : public OpConversionPattern<vector::ToElementsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::ToElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getSource().getType();
    if (sourceType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "not a rank-0 vector");

    // For rank-0 vector, the adapted source is already the scalar.
    // Replace all uses of the single result with the scalar source.
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};
} // namespace

void mlir::aster::populateVectorLegalizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ExtractOpToElements, InsertOpToFromElements,
               ExtractStridedSliceOpToElements, InsertStridedSliceOpToElements,
               StoreOpDropLeadingUnitDims, FromElementsDropLeadingUnitDims>(
      patterns.getContext(), benefit);
  vector::ToElementsOp::getCanonicalizationPatterns(patterns,
                                                    patterns.getContext());
  vector::FromElementsOp::getCanonicalizationPatterns(patterns,
                                                      patterns.getContext());
  vector::ShapeCastOp::getCanonicalizationPatterns(patterns,
                                                   patterns.getContext());
}

void mlir::aster::populateVectorTypeLegalizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<FromElementsRank0Pattern, ToElementsRank0Pattern,
               BroadcastOpToFromElements>(patterns.getContext(), benefit);
}
