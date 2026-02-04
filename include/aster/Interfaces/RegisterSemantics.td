//===-- RegisterSemantics.td - Register semantics -------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines aster register semantics.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_REGISTERSEMANTICS_TD
#define ASTER_REGISTERSEMANTICS_TD

include "aster/Support/EnumBase.td"

//===----------------------------------------------------------------------===//
// Aster register semantics.
//===----------------------------------------------------------------------===//

def RegisterSemantics : AutoI8EnumCases<"RegisterSemantics",
  "Register semantics", [
    PrettyEnumCaseInfo<"allocated">,
    PrettyEnumCaseInfo<"unallocated">,
    PrettyEnumCaseInfo<"value">,
  ]> {
  let cppNamespace = "::mlir::aster";
}

#endif // ASTER_REGISTERSEMANTICS_TD
