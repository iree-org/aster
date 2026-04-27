//===- TargetAttr.td - Target attribute interface ----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetAttr attribute interface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_TARGETATTR_TD
#define ASTER_INTERFACES_TARGETATTR_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TargetAttr interface
//===----------------------------------------------------------------------===//

def TargetAttrInterface : AttrInterface<"TargetAttrInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for attributes that identify a compilation target.
  }];
  let methods = [
    InterfaceMethod<
      "Get the specific target.",
      "::mlir::aster::TargetArch", "getTarget"
    >,
    InterfaceMethod<
      "Get the target family.",
      "::mlir::aster::TargetFamily", "getTargetFamily"
    >
  ];
}

#endif // ASTER_INTERFACES_TARGETATTR_TD
