//===-- AsterUtilsEnums.td - AsterUtils enumerations -------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTERUTILS_ENUMS
#define ASTERUTILS_ENUMS

include "mlir/IR/Utils.td"
include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// GPU dimension identifier.
//===----------------------------------------------------------------------===//

def DimEnum : AutoI32EnumCases<"Dim",
  "GPU dimension identifier", [
    PrettyEnumCaseInfo<"x">,
    PrettyEnumCaseInfo<"y">,
    PrettyEnumCaseInfo<"z">,
  ]> {
  let cppNamespace = "::mlir::aster::aster_utils";
}

#endif // ASTERUTILS_ENUMS
