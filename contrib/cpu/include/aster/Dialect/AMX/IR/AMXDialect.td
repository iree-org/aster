// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_AMX_IR_AMXDIALECT_TD
#define ASTER_DIALECT_AMX_IR_AMXDIALECT_TD

include "mlir/IR/OpBase.td"

def AMX_Dialect : Dialect {
  let name = "amx";
  let summary = "Minimal x86 AMX dialect (tile ops, asm emission only)";
  let cppNamespace = "::mlir::aster::amx";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    private:
      void registerTypes();
  }];
}

class AMX_Type<string name, string mn, list<Trait> traits = []>
    : TypeDef<AMX_Dialect, name, traits> { let mnemonic = mn; }

class AMX_Op<string mnemonic, list<Trait> traits = []>
    : Op<AMX_Dialect, mnemonic, traits>;

#endif // ASTER_DIALECT_AMX_IR_AMXDIALECT_TD
