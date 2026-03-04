//===-- HazardAttrInterface.td - Hazard attribute interface -----*- tablegen -*-===//
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

def HazardAttrInterface : AttrInterface<"HazardAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for attributes that define hazards. This attribute models a hazard
    that can be raised by an operation or operand, and an instruction that can
    trigger the hazard.
  }];
  let methods = [
    InterfaceMethod<[{
      Check if the hazard is triggered by the given instruction.
    }],
    "bool", "isHazardTriggered",
    (ins "const ::mlir::aster::amdgcn::Hazard &":$hazard,
         "::mlir::aster::amdgcn::AMDGCNInstOpInterface":$instOp)
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

#endif // ASTER_AMDGCN_HAZARDATTRINTERFACE_TD
