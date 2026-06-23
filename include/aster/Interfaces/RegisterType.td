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
      }],
      "int64_t", "getSizeInBits"
    >,
    InterfaceMethod<[{
        Returns the properties of the register type.
      }],
      "::mlir::aster::RegisterProps", "getProps"
    >,
    InterfaceMethod<[{
        Returns the composite register type formed by combining this register
        with `types`. `alignment` controls the range alignment (nullopt means
        natural). Returns nullptr on failure.
      }],
      "::mlir::aster::RegisterTypeInterface", "getCompositeType",
      (ins "::mlir::TypeRange":$types,
           "std::optional<int16_t>":$alignment,
           "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
    >,
    InterfaceMethod<[{
        Splits this composite register into its constituent single-register
        parts. Returns failure if not splittable.
      }],
      "::mlir::LogicalResult", "getSplitType",
      (ins "::llvm::SmallVectorImpl<::mlir::aster::RegisterTypeInterface> &":$regs,
           "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)
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

    /// This method returns a clone of the register type with allocated semantics.
    ::mlir::aster::RegisterTypeInterface getAsAllocated(int16_t reg) const {
      return $_type.cloneRegisterType($_type.getAsRange().getAsAllocatedRange(reg));
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

    /// Returns true if two allocated register types have overlapping index ranges.
    bool allocatedRangesOverlap(::mlir::aster::RegisterTypeInterface other) const {
      if (!other || !$_type.hasAllocatedSemantics() ||
          !other.hasAllocatedSemantics())
        return false;
      return $_type.getAsRange().overlaps(other.getAsRange());
    }

    /// Expands composite types and register ranges into constituent single-register
    /// types. Returns failure if a composite part cannot be split further.
    ::mlir::LogicalResult getUnderlyingRegisterTypes(
        ::llvm::SmallVectorImpl<::mlir::aster::RegisterTypeInterface> &regs)
        const {
      if ($_type.isRegisterRange()) {
        ::mlir::aster::RegisterRange range = $_type.getAsRange();
        regs.reserve(regs.size() + range.size());
        for (int16_t i = 0; i < range.size(); ++i)
          regs.push_back(
              $_type.cloneRegisterType(range.begin().getWithOffset(i)));
        return ::mlir::success();
      }
      ::llvm::SmallVector<::mlir::aster::RegisterTypeInterface, 4> parts;
      if (::mlir::succeeded($_type.getSplitType(parts, nullptr))) {
        for (::mlir::aster::RegisterTypeInterface part : parts)
          if (::mlir::failed(part.getUnderlyingRegisterTypes(regs)))
            return ::mlir::failure();
        return ::mlir::success();
      }
      regs.push_back($_type);
      return ::mlir::success();
    }
  }];
}

#endif // ASTER_REGISTERTYPEINTERFACES
