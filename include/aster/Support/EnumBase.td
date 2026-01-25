//===- EnumBase.td - enum helpers --------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tablegen enum helpers.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_SUPPORT_ENUMBASE_TD
#define ASTER_SUPPORT_ENUMBASE_TD

include "mlir/IR/EnumAttr.td"

class EnumCaseInfo<string sym, string id = ""> {
  string symbol = sym;
  string identifier = !if(!empty(id), !toupper(sym), id);
  string description = "";
}

class PrettyEnumCaseInfo<string sym, string id = "">:
  EnumCaseInfo<sym, !if(!empty(id), snakeCaseToCamelCase<sym>.ret, id)>;

class _AutoEnumCase<EnumCaseInfo info, int c, int bitwidth>
    : EnumCase<info.identifier, c, info.symbol, bitwidth> {}

class _EnumCases<list<EnumCaseInfo> caseList, int bitwidth, int start = 0> {
  list<_AutoEnumCase> cases =
      !foreach(i, !range(caseList),
               _AutoEnumCase<caseList[i], !add(i, start), bitwidth>);
}

class AutoEnumCases<string name, string summary, list<EnumCaseInfo> cases,
                    int bitwidth, int start = 0>
    : IntEnum<name, summary, _EnumCases<cases, bitwidth, start>.cases,
              bitwidth> {}

class AutoI8EnumCases<string name, string summary, list<EnumCaseInfo> cases,
                       int start = 0>
    : IntEnum<name, summary, _EnumCases<cases, 8, start>.cases, 8> {}

class AutoI32EnumCases<string name, string summary, list<EnumCaseInfo> cases,
                       int start = 0>
    : IntEnum<name, summary, _EnumCases<cases, 32, start>.cases, 32> {}

class AutoI64EnumCases<string name, string summary, list<EnumCaseInfo> cases,
                       int start = 0>
    : IntEnum<name, summary, _EnumCases<cases, 64, start>.cases, 64> {}

#endif // ASTER_SUPPORT_ENUMBASE_TD
