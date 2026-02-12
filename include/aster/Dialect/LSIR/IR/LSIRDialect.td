//===- LSIRDialect.td - LSIR dialect -----------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LSIR_IR_LSIRDIALECT_TD
#define ASTER_DIALECT_LSIR_IR_LSIRDIALECT_TD

include "mlir/IR/OpBase.td"
include "aster/IR/InstBase.td"
include "aster/Interfaces/InstOpInterface.td"
include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// LSIR dialect definition
//===----------------------------------------------------------------------===//

def LSIR_Dialect : Dialect {
  let name = "lsir";
  let summary = "LSIR high-level dialect";
  let description = [{
    The LSIR dialect provides high-level operations for modeling a
    load-store register architecture.
  }];
  let cppNamespace = "::mlir::aster::lsir";
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

class LSIR_Attr<string name, string typeMnemonic, list<Trait> traits = []>
    : AttrDef<LSIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class LSIR_EnumAttr<string name, EnumInfo enumInfo, list<Trait> traits = []>
    : EnumAttr<LSIR_Dialect, enumInfo, name, traits> {
}

class LSIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LSIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class LSIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<LSIR_Dialect, mnemonic, traits>;

class LSIR_InstOp<string mnemonic, list<Trait> traits = []>
    : InstOperation<LSIR_Dialect, mnemonic, [InstOpInterface] # traits> {
  let genInstVerifier = 0;
}

#endif // ASTER_DIALECT_LSIR_IR_LSIRDIALECT_TD
