//===-- HazardAttrInterface.td - Hazard attribute interfaces ---*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_HAZARDATTRINTERFACE_TD
#define ASTER_AMDGCN_HAZARDATTRINTERFACE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"
include "aster/Interfaces/InstOpInterface.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.td"

def HazardRaiserAttrInterface : AttrInterface<"HazardRaiserAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for attributes that define hazards raised by instructions.
    Populates the hazards vector with hazards raised by the given instruction.
  }];
  let methods = [
    InterfaceMethod<[{
      Determine whether populateHazardsFor should be called based on the
      instruction interface and ISA version.
    }],
    "bool", "matchInst",
    (ins "::mlir::aster::amdgcn::AMDGCNInstOpInterface":$instOp,
         "::mlir::aster::amdgcn::ISAVersion":$isaVer)
    >,
    InterfaceMethod<[{
      Populate the hazards vector with hazards raised by the given instruction.
    }],
    "void", "populateHazardsFor",
    (ins "::mlir::aster::amdgcn::AMDGCNInstOpInterface":$instOp,
         "::llvm::SmallVectorImpl<::mlir::aster::amdgcn::Hazard> &":$hazards)
    >
  ];
}

def HazardCheckerAttrInterface : AttrInterface<"HazardCheckerAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for attributes that define when hazards are triggered.
    Checks if the hazard is triggered by the given instruction.
  }];
  let methods = [
    InterfaceMethod<[{
      Check if the hazard is triggered by the given instruction.
    }],
    "bool", "isHazardTriggered",
    (ins "const ::mlir::aster::amdgcn::Hazard &":$hazard,
         "::mlir::aster::amdgcn::AMDGCNInstOpInterface":$instOp)
    >
  ];
}

#endif // ASTER_AMDGCN_HAZARDATTRINTERFACE_TD
