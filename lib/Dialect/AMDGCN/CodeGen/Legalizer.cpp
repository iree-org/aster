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
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"

#include "aster/Transforms/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
      patterns, 10, [](vector::UnrollVectorOptions &unrollOptions) {
        unrollOptions.setFilterConstraint([](Operation *op) -> LogicalResult {
          return success(OpTrait::hasElementwiseMappableTraits(op));
        });

        // Unroll to scalar: empty native shape means scalar.
        unrollOptions.setNativeShapeFn(
            [](Operation *op) -> std::optional<SmallVector<int64_t>> {
              if (op->getNumResults() == 0)
                return std::nullopt;
              auto vecType = dyn_cast<VectorType>(op->getResult(0).getType());
              if (!vecType)
                return std::nullopt;
              return SmallVector<int64_t>{};
            });
      });

  // Unroll load operations.
  populateUnrollPatterns(
      patterns, 10, [](vector::UnrollVectorOptions &unrollOptions) {
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
      patterns, 10, [](vector::UnrollVectorOptions &unrollOptions) {
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

  // memref.view(memref.alloca) -> amdgcn.alloc_lds + get_lds_offset +
  // byte_shift. Only matches when the alloca is in shared (workgroup)
  // memory space, as set by transform.structured.promote with
  // memory_space = #gpu.address_space<workgroup>.
  // High benefit so this fires before upstream expandStridedMetadata
  // which crashes on memref.view with non-default memory space.
  patterns.add(
      +[](memref::ViewOp viewOp, PatternRewriter &rewriter) -> LogicalResult {
        auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>();
        if (!allocaOp)
          return failure();
        auto allocaType = allocaOp.getMemref().getType();
        if (!allocaType.hasStaticShape() ||
            !allocaType.getElementType().isInteger(8))
          return failure();
        // Only convert shared/workgroup memory to LDS.
        auto memSpace = allocaType.getMemorySpace();
        if (!memSpace)
          return failure();
        // Accept gpu.address_space<workgroup> (printed as integer 3)
        // or any integer attr with value 3.
        if (auto intAttr = dyn_cast<IntegerAttr>(memSpace)) {
          if (intAttr.getInt() != 3)
            return failure();
        } else if (auto gpuAttr = dyn_cast<gpu::AddressSpaceAttr>(memSpace)) {
          if (gpuAttr.getValue() != gpu::AddressSpace::Workgroup)
            return failure();
        } else {
          return failure();
        }
        int64_t sizeBytes = allocaType.getNumElements();
        Location loc = viewOp.getLoc();
        // alloc_lds + get_lds_offset -> index (base LDS byte offset).
        auto ldsAlloc = amdgcn::AllocLDSOp::create(
            rewriter, loc, /*dynamic_size=*/Value(), sizeBytes,
            /*alignment=*/16, /*offset=*/IntegerAttr{});
        auto ldsOffset = amdgcn::GetLDSOffsetOp::create(
            rewriter, loc, rewriter.getIndexType(), ldsAlloc);
        // The view's result type is memref<MxNxT>. Replace with a
        // reinterpret_cast from the LDS offset (treated as a base pointer).
        // The byte_shift from the view op is added to the LDS offset.
        Value base = ldsOffset.getResult();
        Value shift = viewOp.getByteShift();
        Value addr = rewriter.create<arith::AddIOp>(loc, base, shift);
        // Cast the LDS index to the view's memref result type so
        // downstream extract_strided_metadata can decompose it.
        rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
            viewOp, viewOp.getResult().getType(), addr);
        return success();
      },
      PatternBenefit(20));
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
