//===- AMDGCNAttrs.td - AMDGCN attrs ------------------------*- tablegen-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect enum attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_ENUMATTRS_TD
#define ASTER_AMDGCN_ENUMATTRS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNAttrInterfaces.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"

//===----------------------------------------------------------------------===//
// Enum attributes
//===----------------------------------------------------------------------===//

def RegisterKindAttr : AMDGCN_EnumAttr<"RegisterKind", "reg_kind", RegisterKind> {
  let summary = "AMDGCN register kind attribute";
}

def WorkitemIDModeAttr : AMDGCN_EnumAttr<"WorkitemIDMode", "workitem_id_mode", WorkitemIDMode> {
  let summary = "AMDGCN VGPR workitem ID mode attribute";
  let defaultValue = "WorkitemIDMode::X";
}

def FloatRoundModeAttr : AMDGCN_EnumAttr<"FloatRoundMode", "float_round_mode", FloatRoundMode> {
  let summary = "AMDGCN floating point rounding mode attribute";
  let defaultValue = "FloatRoundMode::NearEven";
}

def FloatDenormModeAttr : AMDGCN_EnumAttr<"FloatDenormMode", "float_denorm_mode", FloatDenormMode> {
  let summary = "AMDGCN floating point denorm mode attribute";
  let defaultValue = "FloatDenormMode::SrcDst";
}

def DimAttr : AMDGCN_EnumAttr<"Dim", "dim", DimEnum> {
  let summary = "AMDGCN dimension identifier attribute";
  let assemblyFormat = "$value";
}

#endif // ASTER_AMDGCN_ENUMATTRS_TD
