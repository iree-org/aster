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
    ResourceTypeInterface,
  ]> {
  let description = [{
    This interface defines a common API for interacting with register types.
  }];
  let cppNamespace = "::mlir::aster";
  let methods = [
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
        This method returns the size in bits of the register(s).
        Each register is 32 bits. Returns std::nullopt if the type
        is not a valid register type.
      }],
      "std::optional<int64_t>", "getSizeInBits", (ins), [{}], [{
        return $_type.getAsRange().size() * 32;
      }]
    >
  ];
  let extraTraitClassDeclaration = [{
    bool isAllocatable() const {
      return !$_type.hasAllocatedSemantics();
    }
  }];
  let extraSharedClassDeclaration = [{
    /// This method returns the semantics of the register type.
    ::mlir::aster::RegisterSemantics getSemantics() const {
      return $_type.getAsRange().getSemantics();
    }

    /// This method returns true if the register type has value semantics.
    bool hasValueSemantics() const {
      return getSemantics() == ::mlir::aster::RegisterSemantics::Value;
    }

    /// This method returns true if the register type has unallocated semantics.
    bool hasUnallocatedSemantics() const {
      return getSemantics() == ::mlir::aster::RegisterSemantics::Unallocated;
    }

    /// This method returns true if the register type has allocated semantics.
    bool hasAllocatedSemantics() const {
      return getSemantics() == ::mlir::aster::RegisterSemantics::Allocated;
    }

    /// This method returns a clone of the register type with unallocated semantics.
    ::mlir::aster::RegisterTypeInterface getAsUnallocated() const {
      return $_type.cloneRegisterType($_type.getAsRange().getAsUnallocatedRange());
    }

    /// This method returns a clone of the register type with value semantics.
    ::mlir::aster::RegisterTypeInterface getAsValue() const {
      return $_type.cloneRegisterType($_type.getAsRange().getAsValueRange());
    }

    /// Returns true if this register type represents a range of registers.
    bool isRegisterRange() const {
      return $_type.getAsRange().size() > 1;
    }

    /// This method returns a clone of the register type using the provided
    /// register as the base of the range.
    ::mlir::aster::RegisterTypeInterface cloneRegisterType(::mlir::aster::Register reg) const {
      return $_type.cloneRegisterType(::mlir::aster::RegisterRange(reg, 1));
    }
  }];
}

#endif // ASTER_REGISTERTYPEINTERFACES
