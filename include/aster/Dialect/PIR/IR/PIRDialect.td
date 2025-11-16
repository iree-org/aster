//===- PIRDialect.td - PIR dialect -------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_PIR_IR_PIRDIALECT_TD
#define ASTER_DIALECT_PIR_IR_PIRDIALECT_TD

include "mlir/IR/OpBase.td"
include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// PIR dialect definition
//===----------------------------------------------------------------------===//

def PIR_Dialect : Dialect {
  let name = "pir";
  let summary = "PIR high-level dialect";
  let description = [{
    The PIR dialect provides high-level operations for modeling a
    load-store register architecture.
  }];
  let cppNamespace = "::mlir::aster::pir";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
  let extraClassDeclaration = [{
    private:
      void registerTypes();
      void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class PIR_Attr<string name, string typeMnemonic, list<Trait> traits = []>
    : AttrDef<PIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class PIR_EnumAttr<string name, EnumInfo enumInfo, list<Trait> traits = []>
    : EnumAttr<PIR_Dialect, enumInfo, name, traits> {
  let assemblyFormat = "`<` $value `>`";
}

class PIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class PIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<PIR_Dialect, mnemonic, traits>;

#endif // ASTER_DIALECT_PIR_IR_PIRDIALECT_TD
