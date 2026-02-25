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
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::aster;

void mlir::aster::amdgcn::populateAMDGPULegalizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // TODO: vector unrolling + vector legalization patterns.
  memref::populateExpandOpsPatterns(patterns);
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateExpandStridedMetadataPatterns(patterns);
}

void mlir::aster::amdgcn::populateAMDGPUTypeLegalizationPatterns(
    TypeConverter &converter, ConversionTarget &target,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // TODO: vector type legalization patterns.
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalDialect<arith::ArithDialect, ptr::PtrDialect,
                         affine::AffineDialect>();
  populateArithConversionPatterns(converter, target, patterns);
  populateScfConversionPatterns(converter, target, patterns);
  populatePtrConversionPatterns(converter, target, patterns);
  populateFuncConversionPatterns(converter, target, patterns);
  populateMemOpsConversionPatterns(converter, patterns);
}
