//===-- RegisterType.td - Register type interfaces ------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines aster register-like type interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_REGISTERTYPEINTERFACES
#define ASTER_REGISTERTYPEINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "aster/Interfaces/ResourceInterfaces.td"

//===----------------------------------------------------------------------===//
// Aster register-like type interface.
//===----------------------------------------------------------------------===//

def RegisterTypeInterface : TypeInterface<"RegisterTypeInterface", [
    ResourceTypeInterface
  ]> {
  let description = [{
    This interface defines a common API for interacting with register types.
  }];
  let cppNamespace = "::mlir::aster";
  let methods = [
    InterfaceMethod<[{
        This method returns whether the register is relocatable.
      }],
      "bool", "isRelocatable"
    >,
    InterfaceMethod<[{
        This method returns whether the type is a register range.
      }],
      "bool", "isRegisterRange"
    >,
    InterfaceMethod<[{
        This method returns the register's range.
      }],
      "::mlir::aster::RegisterRange", "getAsRange"
    >,
    InterfaceMethod<[{
        This method returns a clone of the register.
      }],
      "::mlir::aster::RegisterTypeInterface", "cloneRegisterType",
      (ins "::mlir::aster::RegisterRange":$range)
    >,
    InterfaceMethod<[{
        This method returns a clone of the register.
      }],
      "::mlir::aster::RegisterTypeInterface", "cloneRegisterType",
      (ins "::mlir::aster::Register":$reg)
    >,
    InterfaceMethod<[{
        This method returns the size in bytes of the register(s).
        Each register is 4 bytes (32-bit). Returns std::nullopt if the type
        is not a valid register type.
      }],
      "std::optional<int64_t>", "getSizeInBytes", (ins), [{}], [{
        return $_type.getAsRange().size() * 4;
      }]
    >
  ];
  let extraTraitClassDeclaration = [{
    bool isAllocatable() const {
      return $_type.isRelocatable();
    }
  }];
}

#endif // ASTER_REGISTERTYPEINTERFACES
