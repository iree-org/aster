//===- Init.cpp -----------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Init.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/PIR/IR/PIRDialect.h"
#include "aster/Dialect/PIR/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"

using namespace mlir;

void mlir::aster::initDialects(DialectRegistry &registry) {
  registry.insert<amdgcn::AMDGCNDialect>();
  registry.insert<pir::PIRDialect>();
}

void mlir::aster::initUpstreamMLIRDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<ptr::PtrDialect>();
}

void mlir::aster::asterInitDialects(MlirDialectRegistry registry) {
  initDialects(*unwrap(registry));
}

void mlir::aster::registerPasses() {
  amdgcn::registerAMDGCNPasses();
  pir::registerPIRPasses();
}
