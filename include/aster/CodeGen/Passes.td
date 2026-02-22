//===- Passes.td - CodeGen passes --------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_CODEGEN_PASSES_TD
#define ASTER_CODEGEN_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// CodeGen
//===----------------------------------------------------------------------===//

def CodeGen : InterfacePass<"aster-codegen", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert high-level dialects to the low-level dialects";
}

//===----------------------------------------------------------------------===//
// Legalizer
//===----------------------------------------------------------------------===//

def Legalizer : Pass<"aster-legalizer"> {
  let summary = "Legalize operations for code generation";
  let description = [{
    This pass legalizes operations to forms that are supported by the
    target architecture. It handles vector type conversions, operation
    lowering, and memory access transformations as needed.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::ptr::PtrDialect",
    "mlir::vector::VectorDialect",
    "mlir::DLTIDialect",
    "aster::amdgcn::AMDGCNDialect"
  ];
}

#endif // ASTER_CODEGEN_PASSES_TD
