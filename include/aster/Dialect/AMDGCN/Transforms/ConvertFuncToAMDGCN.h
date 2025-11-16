//===- ConvertFuncToAMDGCN.h - Func to AMDGCN conversion ---------*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the function to convert Func dialect operations to AMDGCN
// dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_DIALECT_AMDGCN_TRANSFORMS_CONVERTFUNCTOAMDGCN_H
#define AMDGCN_DIALECT_AMDGCN_TRANSFORMS_CONVERTFUNCTOAMDGCN_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir::aster {
namespace amdgcn {
class KernelOp;
} // namespace amdgcn
} // namespace mlir::aster

namespace mlir {

/// Convert a func::FuncOp to an amdgcn::KernelOp.
/// Only supports ptr and POD (integer/float) arguments that are translated
/// one-to-one. Index type is not supported. The function must be within an
/// amdgcn::ModuleOp. Return failure if failed to convert.
FailureOr<aster::amdgcn::KernelOp>
convertFuncOpToAMDGCNKernel(func::FuncOp funcOp, RewriterBase &rewriter);

} // namespace mlir

#endif // AMDGCN_DIALECT_AMDGCN_TRANSFORMS_CONVERTFUNCTOAMDGCN_H
