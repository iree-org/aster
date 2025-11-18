//===- AsterUtilsDialect.td - AsterUtils dialect -----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSDIALECT_TD
#define ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSDIALECT_TD

include "mlir/IR/OpBase.td"
include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// AsterUtils dialect definition
//===----------------------------------------------------------------------===//

def AsterUtils_Dialect : Dialect {
  let name = "aster_utils";
  let summary = "Aster utility dialect";
  let description = [{
    The AsterUtils dialect provides utility operations for the Aster compiler.
  }];
  let cppNamespace = "::mlir::aster::aster_utils";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    private:
      void registerTypes();
      void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class AsterUtils_Attr<string name, string typeMnemonic, list<Trait> traits = []>
    : AttrDef<AsterUtils_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class AsterUtils_EnumAttr<string name, EnumInfo enumInfo, list<Trait> traits = []>
    : EnumAttr<AsterUtils_Dialect, enumInfo, name, traits> {
}

class AsterUtils_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AsterUtils_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class AsterUtils_Op<string mnemonic, list<Trait> traits = []>
    : Op<AsterUtils_Dialect, mnemonic, traits>;

#endif // ASTER_DIALECT_ASTERUTILS_IR_ASTERUTILSDIALECT_TD
