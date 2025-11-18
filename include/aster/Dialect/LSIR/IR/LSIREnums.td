//===-- LSIREnums.td - LSIR dialect enumerations -----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LSIR_ENUMS
#define LSIR_ENUMS

include "mlir/IR/Utils.td"
include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// LSIR dim identifier.
//===----------------------------------------------------------------------===//

def DimEnum : AutoI32EnumCases<"Dim",
  "LSIR dimension identifier", [
    PrettyEnumCaseInfo<"x">,
    PrettyEnumCaseInfo<"y">,
    PrettyEnumCaseInfo<"z">,
  ]> {
  let cppNamespace = "::mlir::aster::lsir";
}

#endif // LSIR_ENUMS
