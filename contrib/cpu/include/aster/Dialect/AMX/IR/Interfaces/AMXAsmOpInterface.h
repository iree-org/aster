// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_AMX_IR_INTERFACES_AMXASMOPINTERFACE_H
#define ASTER_DIALECT_AMX_IR_INTERFACES_AMXASMOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "aster/Dialect/AMX/IR/Interfaces/AMXAsmOpInterface.h.inc"

namespace mlir::aster::amx {

/// Return the physical x86_64 register name encoded in an `!x86.gpr<reg>`
/// or `!amx.tile<"...", ...>` value type.
::llvm::StringRef getPhysicalRegisterName(::mlir::Value v);

} // namespace mlir::aster::amx

#endif // ASTER_DIALECT_AMX_IR_INTERFACES_AMXASMOPINTERFACE_H
