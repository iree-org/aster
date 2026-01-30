//===- Legalizer.cpp - AMDGCN legalization patterns ----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/Legalizer.h"
#include "aster/Dialect/AMDGCN/CodeGen/CodeGen.h"

#include "aster/Transforms/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;

static void populateUnrollPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit,
    llvm::function_ref<void(vector::UnrollVectorOptions &)> configFn) {
  // Configure vector unrolling options.
  vector::UnrollVectorOptions unrollOptions;
  // Allow customization of unroll options.
  configFn(unrollOptions);
  // Add vector unroll patterns for elementwise operations.
  vector::populateVectorUnrollPatterns(patterns, unrollOptions, benefit);
}

void mlir::aster::amdgcn::populateAMDGPULegalizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {

  // Unroll elementwise vector operations to scalars.
  populateUnrollPatterns(
      patterns, 10, +[](vector::UnrollVectorOptions &unrollOptions) {
        // Only unroll operations with the ElementwiseMappable trait.
        unrollOptions.setFilterConstraint([](Operation *op) -> LogicalResult {
          return success(OpTrait::hasElementwiseMappableTraits(op));
        });

        // Set the native shape to unroll to (e.g., scalar or small vectors).
        unrollOptions.setNativeShapeFn(
            [](Operation *op) -> std::optional<SmallVector<int64_t>> {
              // Unroll to scalar (shape of 1 for each dimension).
              if (op->getNumResults() == 0)
                return std::nullopt;
              auto vecType = dyn_cast<VectorType>(op->getResult(0).getType());
              if (!vecType)
                return std::nullopt;
              return SmallVector<int64_t>(0);
            });
      });

  // Unroll load operations.
  populateUnrollPatterns(
      patterns, 10, +[](vector::UnrollVectorOptions &unrollOptions) {
        // Only unroll operations with the ElementwiseMappable trait.
        unrollOptions.setFilterConstraint([](Operation *op) -> LogicalResult {
          return success(isa<vector::TransferReadOp, vector::LoadOp>(op));
        });

        // Set the native shape to unroll to.
        unrollOptions.setNativeShapeFn(
            [](Operation *op) -> std::optional<SmallVector<int64_t>> {
              VectorType vecType;
              if (auto xferOp = dyn_cast<vector::TransferReadOp>(op))
                vecType = xferOp.getType();
              else if (auto loadOp = dyn_cast<vector::LoadOp>(op))
                vecType = loadOp.getType();
              if (!vecType || vecType.getRank() < 2)
                return std::nullopt;

              return SmallVector<int64_t>(1, vecType.getShape().back());
            });
      });

  // Unroll store operations. NOTE: this has to be separate from load unrolling
  // because unrolling stores produces extract_strided_slice which have tighter
  // verification rules than insert.
  populateUnrollPatterns(
      patterns, 10, +[](vector::UnrollVectorOptions &unrollOptions) {
        // Only unroll operations with the ElementwiseMappable trait.
        unrollOptions.setFilterConstraint([](Operation *op) -> LogicalResult {
          return success(isa<vector::TransferWriteOp, vector::StoreOp>(op));
        });

        // Set the native shape to unroll to.
        unrollOptions.setNativeShapeFn(
            [](Operation *op) -> std::optional<SmallVector<int64_t>> {
              VectorType vecType;
              if (auto xferOp = dyn_cast<vector::TransferWriteOp>(op))
                vecType = xferOp.getVectorType();
              else if (auto storeOp = dyn_cast<vector::StoreOp>(op))
                vecType = storeOp.getVectorType();

              if (!vecType || vecType.getRank() < 2)
                return std::nullopt;

              SmallVector<int64_t> shape(vecType.getRank() - 1, 1);
              shape.push_back(vecType.getShape().back());
              return shape;
            });
      });
  vector::populateDropUnitDimWithShapeCastPatterns(patterns, 5);
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns, 4);
  populateVectorLegalizationPatterns(patterns, 2);
  // TODO: Fix upstream so these receive a benefit parameter.
  memref::populateExpandOpsPatterns(patterns);
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateExpandStridedMetadataPatterns(patterns);
}

void mlir::aster::amdgcn::populateAMDGPUTypeLegalizationPatterns(
    TypeConverter &converter, ConversionTarget &target,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  converter.addConversion([](VectorType type) -> Type {
    if (int64_t rank = type.getRank();
        rank == 0 || (rank == 1 && type.getDimSize(0) == 1)) {
      return type.getElementType();
    }
    return type;
  });
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalDialect<arith::ArithDialect, ptr::PtrDialect,
                         affine::AffineDialect>();
  target.addDynamicallyLegalOp<vector::FromElementsOp, vector::ToElementsOp>(
      [&](Operation *op) { return converter.isLegal(op); });
  populateArithConversionPatterns(converter, target, patterns);
  populateScfConversionPatterns(converter, target, patterns);
  populateVectorTypeLegalizationPatterns(patterns, benefit);
  populatePtrConversionPatterns(converter, target, patterns);
  populateFuncConversionPatterns(converter, target, patterns);
  populateMemOpsConversionPatterns(converter, patterns);
}
