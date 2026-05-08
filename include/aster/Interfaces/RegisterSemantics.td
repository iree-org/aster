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
// This file defines aster register semantics and properties.
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

//===----------------------------------------------------------------------===//
// Aster register properties.
//===----------------------------------------------------------------------===//

def RegisterProps : I8BitEnum<"RegisterProps",
  "Register type properties", [
    I8BitEnumCaseNone<"None", "none">,
    I8BitEnumCase<"IsGeneralPurpose", 1, "is_general_purpose">,
    I8BitEnumCase<"IsComposite", 2, "is_composite">,
    I8BitEnumCase<"AcceptsComposite", 4, "accepts_composite">,
    I8BitEnumCase<"AcceptsValueSemantics", 8, "accepts_value_semantics">,
    I8BitEnumCase<"AcceptsUnallocatedSemantics", 16, "accepts_unallocated_semantics">,
    I8BitEnumCase<"AcceptsAllocatedSemantics", 32, "accepts_allocated_semantics">
  ]> {
  let cppNamespace = "::mlir::aster";
}

#endif // ASTER_REGISTERSEMANTICS_TD
