// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_INTERFACES_X86ISAOPINTERFACE_TD
#define ASTER_DIALECT_X86_IR_INTERFACES_X86ISAOPINTERFACE_TD

include "mlir/IR/OpBase.td"
include "aster/Dialect/X86/IR/X86Enums.td"

def X86IsaOpInterface : OpInterface<"X86IsaOpInterface"> {
  let description = [{
    Interface for x86 ops that announce the minimum x86 ISA extension
    level they require to execute. A function-level
    `{x86.isa = #x86.isa<avx512>}` attribute declares the target for
    the enclosing function; the TranslateModule pass verifies that
    every op in the body has `getRequiredIsa()` <= that target.
  }];
  let cppNamespace = "::mlir::aster::x86";
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the minimum x86 ISA level required to execute this op.",
      /*retTy=*/"::mlir::aster::x86::IsaVersion",
      /*methodName=*/"getRequiredIsa",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >
  ];
}

#endif // ASTER_DIALECT_X86_IR_INTERFACES_X86ISAOPINTERFACE_TD
