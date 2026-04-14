// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_AMX_IR_INTERFACES_AMXASMOPINTERFACE_TD
#define ASTER_DIALECT_AMX_IR_INTERFACES_AMXASMOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def AMXAsmOpInterface : OpInterface<"AMXAsmOpInterface"> {
  let description = [{
    Interface for AMX ops that know how to print themselves as one line of
    x86_64 AT&T assembly.
  }];
  let cppNamespace = "::mlir::aster::amx";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{ Emit one line of AT&T asm for this op. }],
      /*retTy=*/"void",
      /*methodName=*/"printAsm",
      /*args=*/(ins "::llvm::raw_ostream &":$os)
    >
  ];
}

#endif // ASTER_DIALECT_AMX_IR_INTERFACES_AMXASMOPINTERFACE_TD
