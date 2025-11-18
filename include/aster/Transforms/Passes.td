//===- Passes.td - Aster passes ----------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TRANSFORMS_PASSES_TD
#define ASTER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ReplaceConstantGPUDims
//===----------------------------------------------------------------------===//

def ReplaceConstantGPUDims : InterfacePass<"aster-replace-constant-gpu-dims", "mlir::aster::ModuleOpInterface"> {
  let summary = "Replace gpu.block_dim and gpu.grid_dim with constants";
  let description = [{
    This pass identifies functions implementing GPUFuncInterface and replaces
    gpu.block_dim and gpu.grid_dim operations with constant values based on
    the block and grid dimension annotations on the function.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::gpu::GPUDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ToIntArith
//===----------------------------------------------------------------------===//

def ToIntArith : InterfacePass<"aster-to-int-arith", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert index operations to integer operations";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];
  let options = [
    Option<"bitwidth", "bitwidth", "unsigned", "32",
           "Index bitwidth to convert to">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::aster::aster_utils::AsterUtilsDialect",
  ];
}

//===----------------------------------------------------------------------===//
// OptimizeArith
//===----------------------------------------------------------------------===//

def OptimizeArith : InterfacePass<"aster-optimize-arith", "mlir::aster::ModuleOpInterface"> {
  let summary = "Optimize arithmetic operations";
}

#endif // ASTER_TRANSFORMS_PASSES_TD
