// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Init.cpp - mlir-air dialect and pass registration ------------------===//

#include "air/Conversion/ConvertToAIRPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Transform/AIRDmaToChannel.h"

// Tablegen-generated per-pass registration for upstream AIR passes.
namespace air_conv_reg {
#define GEN_PASS_REGISTRATION_COPYTODMA
#define GEN_PASS_REGISTRATION_PARALLELTOHERD
#define GEN_PASS_REGISTRATION_PARALLELTOLAUNCH
#define GEN_PASS_REGISTRATION_AIRWRAPFUNCWITHPARALLELPASS
#include "air/Conversion/Passes.h.inc"
} // namespace air_conv_reg

namespace air_xform_reg {
#define GEN_PASS_REGISTRATION_DMATOCHANNEL
#include "air/Transform/Passes.h.inc"
} // namespace air_xform_reg

#include "aster/Init.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
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

std::unique_ptr<Pass> createAirToAMDGCN();
std::unique_ptr<Pass> createConvertLinalgToAMDGCN();
std::unique_ptr<Pass> createConvertMemSpaceToAMDGCN();
void registerPipelines();

void registerAll(DialectRegistry &registry) {
  // AIR dialect.
  registry.insert<xilinx::air::airDialect>();

  // Dialects needed for linalg tiling + transform dialect.
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<transform::TransformDialect>();

  // Bufferization interface models.
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerSubsetOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);

  // Transform dialect extensions.
  bufferization::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);

  // Tiling interface for linalg ops.
  linalg::registerTilingInterfaceExternalModels(registry);

  // Upstream passes.
  bufferization::registerBufferizationPasses();
  registerLinalgPasses();
  memref::registerMemRefPasses();
  transform::registerInterpreterPass();
  transform::registerPreloadLibraryPass();

  // AIR transform ops extension (air.transform.*).
  xilinx::air::registerTransformDialectExtension(registry);

  // Upstream doesn't declare airDialect as a dependent of the transform
  // extension — add it so par_to_herd can create air.herd ops.
  registry.addExtension(
      +[](MLIRContext *ctx, transform::TransformDialect *dialect) {
        ctx->getOrLoadDialect<xilinx::air::airDialect>();
      });

  // Upstream AIR passes (tablegen-generated registration).
  air_conv_reg::registerCopyToDma();   // air-copy-to-dma
  air_conv_reg::registerParallelToHerd(); // air-par-to-herd
  air_conv_reg::registerParallelToLaunch(); // air-par-to-launch
  air_conv_reg::registerAIRWrapFuncWithParallelPass(); // air-wrap-func-with-parallel
  air_xform_reg::registerDmaToChannel(); // air-dma-to-channel

  // Aster-specific passes.
  registerPass([] { return createAirToAMDGCN(); });
  registerPass([] { return createConvertLinalgToAMDGCN(); });
  registerPass([] { return createConvertMemSpaceToAMDGCN(); });

  // mlir-air pipelines.
  registerPipelines();
}

// Register mlir-air dialects/passes into aster's init hook so they
// are available in aster-opt and the Python bindings when linked.
static int _register = (mlir::aster::registerContribDialects(registerAll), 0);


} // namespace mlir::aster::mlir_air
