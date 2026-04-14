// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_AMX_REGISTRATION_H
#define ASTER_DIALECT_AMX_REGISTRATION_H

namespace mlir::aster::amx {

/// Register the `mlir-to-amx-asm` translation with MLIR's global registry.
void registerTranslateToAMXAsm();

} // namespace mlir::aster::amx

#endif // ASTER_DIALECT_AMX_REGISTRATION_H
