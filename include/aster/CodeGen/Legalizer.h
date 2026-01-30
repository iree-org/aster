//===- Legalizer.h - CodeGen legalization patterns --------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_CODEGEN_LEGALIZER_H
#define ASTER_CODEGEN_LEGALIZER_H

#include "aster/Transforms/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ConversionTarget;
class RewritePatternSet;

namespace aster {
//===----------------------------------------------------------------------===//
// MemRefDescriptor
//===----------------------------------------------------------------------===//

/// Helper class to access the decomposed members of a converted memref value.
/// A memref is decomposed into:
///   - aligned ptr
///   - base ptr
///   - offset (as OpFoldResult)
///   - sizes (as OpFoldResults)
///   - strides (as OpFoldResults)
class MemRefDescriptor {
public:
  /// Create a MemRefDescriptor from a memref type and its converted values.
  static FailureOr<MemRefDescriptor> get(MemRefType type, ValueRange values);

  /// Get the converted types for a memref type.
  /// Populates results with: ptr.ptr, dynamic offset index (if dynamic),
  /// dynamic size indices, and dynamic stride indices.
  static LogicalResult convertType(const TypeConverter &converter,
                                   MemRefType type,
                                   SmallVectorImpl<Type> &results);

  /// Get the aligned pointer.
  Value getAlignedPtr() const { return alignedPtr; }

  /// Get the base pointer.
  Value getBasePtr() const { return basePtr; }

  /// Get the offset as an OpFoldResult (Value if dynamic, Attribute if static).
  OpFoldResult getOffset() const { return offset; }

  /// Get the size at dimension i as an OpFoldResult.
  OpFoldResult getSize(int64_t i) const;

  /// Get the stride at dimension i as an OpFoldResult.
  OpFoldResult getStride(int64_t i) const;

  /// Get the number of sizes (i.e., the rank).
  int64_t getNumSizes() const { return numSizes; }

  /// Get the number of strides (i.e., the rank).
  int64_t getNumStrides() const { return sizesAndStrides.size() - numSizes; }

  /// Get the original memref type.
  MemRefType getType() const { return type; }

private:
  /// Construct a descriptor from a memref type and its converted values.
  MemRefDescriptor(MemRefType type) : type(type) {}

  MemRefType type;
  Value alignedPtr;
  Value basePtr;
  OpFoldResult offset;
  SmallVector<OpFoldResult> sizesAndStrides;
  int64_t numSizes = 0;
};

//===----------------------------------------------------------------------===//
// LegalizerTypeConverter
//===----------------------------------------------------------------------===//

/// Type converter for legalizing types during code generation.
/// Converts memref types to a sequence of values:
///   - aligned ptr (ptr.ptr type)
///   - index for dynamic offset (if the offset is dynamic)
///   - index for each dynamic size dimension
///   - index for each dynamic stride
class LegalizerTypeConverter : FuncTypeConverter, public TypeConverter {
public:
  explicit LegalizerTypeConverter(MLIRContext *ctx);
};

//===----------------------------------------------------------------------===//
// Pattern population functions
//===----------------------------------------------------------------------===//

/// Populate patterns for AMDGPU-specific legalization.
void populateAMDGPULegalizationPatterns(RewritePatternSet &patterns,
                                        PatternBenefit benefit = 1);

/// Populate patterns for AMDGPU type legalization using dialect conversion.
void populateAMDGPUTypeLegalizationPatterns(TypeConverter &converter,
                                            ConversionTarget &target,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit = 1);

/// Populate conversion patterns for memory operations (memref.load,
/// vector.load, vector.store, etc.) to ptr dialect operations.
/// Uses the type converter for 1-to-N conversions.
void populateMemOpsConversionPatterns(const TypeConverter &converter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit = 1);

/// Populate patterns to legalize vector operations (extract, insert, broadcast,
/// strided slice) to element-wise operations using ToElementsOp/FromElementsOp.
void populateVectorLegalizationPatterns(RewritePatternSet &patterns,
                                        PatternBenefit benefit = 1);
void populateVectorTypeLegalizationPatterns(RewritePatternSet &patterns,
                                            PatternBenefit benefit = 1);

} // namespace aster
} // namespace mlir

#endif // ASTER_CODEGEN_LEGALIZER_H
