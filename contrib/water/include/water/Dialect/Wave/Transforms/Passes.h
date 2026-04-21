// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_PASSES_H
#define WATER_DIALECT_WAVE_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::aster::amdgcn {
class AMDGCNDialect;
} // namespace mlir::aster::amdgcn
namespace mlir::aster::aster_utils {
class AsterUtilsDialect;
} // namespace mlir::aster::aster_utils
namespace mlir::aster::lsir {
class LSIRDialect;
} // namespace mlir::aster::lsir
namespace mlir::affine {
class AffineDialect;
} // namespace mlir::affine
namespace mlir::arith {
class ArithDialect;
} // namespace mlir::arith
namespace mlir::cf {
class ControlFlowDialect;
} // namespace mlir::cf
namespace mlir {
class DLTIDialect;
} // namespace mlir
namespace mlir::func {
class FuncDialect;
} // namespace mlir::func
namespace mlir::gpu {
class GPUDialect;
} // namespace mlir::gpu
namespace mlir::memref {
class MemRefDialect;
} // namespace mlir::memref
namespace mlir::pdl {
class PDLDialect;
} // namespace mlir::pdl
namespace mlir::pdl_interp {
class PDLInterpDialect;
} // namespace mlir::pdl_interp
namespace mlir::ptr {
class PtrDialect;
} // namespace mlir::ptr
namespace mlir::scf {
class SCFDialect;
} // namespace mlir::scf
namespace mlir::ub {
class UBDialect;
} // namespace mlir::ub
namespace mlir::vector {
class VectorDialect;
} // namespace mlir::vector
namespace wave {

#define GEN_PASS_DECL
#include "water/Dialect/Wave/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "water/Dialect/Wave/Transforms/Passes.h.inc"

} // namespace wave

#endif // WATER_DIALECT_WAVE_TRANSFORMS_PASSES_H
