// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86TYPES_TD
#define ASTER_DIALECT_X86_IR_X86TYPES_TD

include "aster/Dialect/X86/IR/X86Dialect.td"
include "aster/Dialect/X86/IR/X86Enums.td"

def X86_GprType : X86_Type<"Gpr", "gpr"> {
  let summary = "Pre-assigned x86_64 general-purpose register";
  let description = [{
    An x86_64 GPR bound to a specific physical register.

    ```mlir
    !x86.gpr<rdi>
    !x86.gpr<r8>
    ```
  }];
  let parameters = (ins EnumParameter<X86_GprEnum>:$reg);
  let assemblyFormat = "`<` $reg `>`";
}

def X86_AnyGpr : Type<
  CPred<"::llvm::isa<::mlir::aster::x86::GprType>($_self)">,
  "x86 gpr">;

#endif // ASTER_DIALECT_X86_IR_X86TYPES_TD
