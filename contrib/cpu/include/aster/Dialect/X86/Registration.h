// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_REGISTRATION_H
#define ASTER_DIALECT_X86_REGISTRATION_H

namespace mlir::aster::x86 {

/// Register the `mlir-to-x86-asm` translation with MLIR's global registry.
void registerTranslateToX86Asm();

} // namespace mlir::aster::x86

#endif // ASTER_DIALECT_X86_REGISTRATION_H
