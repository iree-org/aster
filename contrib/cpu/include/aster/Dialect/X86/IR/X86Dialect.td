// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86DIALECT_TD
#define ASTER_DIALECT_X86_IR_X86DIALECT_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def X86_Dialect : Dialect {
  let name = "x86";
  let summary = "Minimal x86 register types + scalar/vector ops";
  let description = [{
    Owns the `!x86.gpr<reg>` general-purpose register type, the
    `!x86.avx.xmm`, `!x86.avx2.ymm`, `!x86.avx512.zmm` vector register
    types, and ISA-gated ops.
  }];
  let cppNamespace = "::mlir::aster::x86";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class X86_Type<string name, string mn, list<Trait> traits = []>
    : TypeDef<X86_Dialect, name, traits> {
  let mnemonic = mn;
}

class X86_EnumAttr<string name, string attrMnemonic, EnumInfo enumInfo,
                   list<Trait> traits = []>
    : EnumAttr<X86_Dialect, enumInfo, name, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` $value `>`";
}

#endif // ASTER_DIALECT_X86_IR_X86DIALECT_TD
