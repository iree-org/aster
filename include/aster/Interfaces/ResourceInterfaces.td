//===- ResourceInterfaces.td - Resource Interfaces ----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines resource type and allocation operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_RESOURCEINTERFACES_TD
#define ASTER_INTERFACES_RESOURCEINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ResourceTypeInterface
//===----------------------------------------------------------------------===//

def ResourceTypeInterface : TypeInterface<"ResourceTypeInterface"> {
  let cppNamespace = "::mlir::aster";
  let description = [{
    Interface for types that represent hardware or abstract resources.
    Resources are values that may require special allocation, deallocation,
    or lifetime management.
  }];
  let methods = [
    InterfaceMethod<[{
        This method returns whether the resource is allocatable.
      }],
      "bool", "isAllocatable"
    >,
    InterfaceMethod<[{
        This method returns the underlying resource.
      }],
      "::mlir::aster::Resource*", "getResource"
    >
  ];
}

#endif // ASTER_INTERFACES_RESOURCEINTERFACES_TD
