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

//===----------------------------------------------------------------------===//
// Aster register-like type interface.
//===----------------------------------------------------------------------===//

def RegisterTypeInterface : TypeInterface<"RegisterTypeInterface"> {
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
        This method returns the register's kind as an attribute.
      }],
      "::mlir::Attribute", "getRegisterKind"
    >,
    InterfaceMethod<[{
        This method returns the register's kind as an int.
      }],
      "int64_t", "getRegisterKindAsInt"
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
    >
  ];
}

#endif // ASTER_REGISTERTYPEINTERFACES
