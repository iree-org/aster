// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86ATTRS_TD
#define ASTER_DIALECT_X86_IR_X86ATTRS_TD

include "aster/Dialect/X86/IR/X86Dialect.td"
include "aster/Dialect/X86/IR/X86Enums.td"

def X86_IsaVersionAttr : X86_EnumAttr<"IsaVersion", "isa", X86_IsaVersionEnum>;

#endif // ASTER_DIALECT_X86_IR_X86ATTRS_TD
