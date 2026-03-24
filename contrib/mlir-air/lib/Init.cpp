// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Init.cpp - mlir-air dialect and pass registration ------------------===//

#include "aster/Init.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::aster::mlir_air {

std::unique_ptr<Pass> createConvertLinalgToAMDGCN();
void registerPipelines();

void registerAll(DialectRegistry &registry) {
  // Dialects needed for linalg tiling + transform dialect.
  registry.insert<linalg::LinalgDialect>();
  registry.insert<transform::TransformDialect>();

  // Transform dialect extensions.
  linalg::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);

  // Tiling interface for linalg ops.
  linalg::registerTilingInterfaceExternalModels(registry);

  // Upstream passes.
  registerLinalgPasses();
  memref::registerMemRefPasses();
  transform::registerInterpreterPass();

  registerPass([] { return createConvertLinalgToAMDGCN(); });

  // mlir-air pipelines.
  registerPipelines();
}

// Register mlir-air dialects/passes into aster's init hook so they
// are available in aster-opt and the Python bindings when linked.
static int _register = (mlir::aster::registerContribDialects(registerAll), 0);

} // namespace mlir::aster::mlir_air
