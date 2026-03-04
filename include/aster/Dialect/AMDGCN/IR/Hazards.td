//===- Hazards.td - AMDGCN hazard definitions -----------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines AMDGCN hazards.
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_HAZARDS_TD
#define ASTER_AMDGCN_HAZARDS_TD

include "mlir/IR/Utils.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/HazardAttrInterface.td"
include "aster/Support/EnumBase.td"

class HazardAttr<string name, string attrMnemonic> :
  AMDGCN_Attr<name, attrMnemonic, [
    DeclareAttrInterfaceMethods<HazardAttrInterface>
  ]> {
}

//===----------------------------------------------------------------------===//
// CDNA3 Hazards
//===----------------------------------------------------------------------===//

def CDNA3_StoreHazard : HazardAttr<"CDNA3StoreHazard", "cdna3_store_hazard"> {
  let summary = "CDNA3 store hazard followed by VALU write";
  let description = [{
    This hazard is raised when a store operation is followed by a VALU instruction
    overwriting the write data of the store.
  }];
  let extraClassDeclaration = [{
    /// The number of waits states required to resolve the hazard.
    static constexpr int32_t requiredWaits = 2;
  }];
}

#endif // ASTER_AMDGCN_HAZARDS_TD
