//===- AMDGCNInsts.td - AMDGCN Instruction definitions -----*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_INSTS_TD
#define AMDGCN_INSTS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNInstBase.td"
include "aster/Dialect/AMDGCN/IR/Instructions/VOP.td"

include "aster/Dialect/AMDGCN/IR/AMDGCNOps.td"

//===----------------------------------------------------------------------===//
// AMDGCN  Enums
//===----------------------------------------------------------------------===//

def EncodingEnum : AutoI32EnumCases<"AMDGCNEncoding",
  "AMDGCN encodings", !instances<Encoding>() # [
    PrettyEnumCaseInfo<"last_enc">
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def InstPropEnum : AutoI32EnumCases<"InstProp",
  "AMDGCN instruction properties", !instances<InstProp>() # [
    PrettyEnumCaseInfo<"last_prop">
  ]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def OpCodeEnum : AutoI64EnumCases<"OpCode", "AMDGCN Operation Codes",
    [InvalidCase] # !instances<AMDInstOpCode>() # [
      PrettyEnumCaseInfo<"last_op_code">
    ]
  > {
  let cppNamespace = "::mlir::aster::amdgcn";
}

#endif // AMDGCN_INSTS_TD
