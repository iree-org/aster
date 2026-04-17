// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_TARGET_TRANSLATETOASM_H
#define ASTER_DIALECT_X86_TARGET_TRANSLATETOASM_H

#include "mlir/IR/Operation.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::aster::x86 {

::llvm::LogicalResult translateToAsm(::mlir::Operation *op,
                                     ::llvm::raw_ostream &os);

} // namespace mlir::aster::x86

#endif // ASTER_DIALECT_X86_TARGET_TRANSLATETOASM_H
