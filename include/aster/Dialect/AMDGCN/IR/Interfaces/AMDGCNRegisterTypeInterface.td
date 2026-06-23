//===-- AMDGCNRegisterTypeInterface.td - AMDGCN register type interface -*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_AMDGCNREGISTERTYPEINTERFACE_TD
#define ASTER_AMDGCN_AMDGCNREGISTERTYPEINTERFACE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"
include "aster/Interfaces/RegisterType.td"

def AMDGCNRegisterTypeInterface : TypeInterface<"AMDGCNRegisterTypeInterface", [
    RegisterTypeInterface
  ]> {
  let description = [{
    Interface for AMDGCN register-like types.
  }];
  let cppNamespace = "::mlir::aster::amdgcn";
  let methods = [
    InterfaceMethod<
      "Get the AMDGCN register kind of this type.",
      "::mlir::aster::amdgcn::RegisterKind", "getRegisterKind"
    >
  ];
  let extraSharedClassDeclaration = [{
    /// Returns true if two allocated AMDGCN register types alias (same kind
    /// and overlapping index ranges).
    bool overlaps(AMDGCNRegisterTypeInterface other) const {
      if (!other || !$_type.hasAllocatedSemantics() ||
          !other.hasAllocatedSemantics())
        return false;
      if ($_type.getRegisterKind() != other.getRegisterKind())
        return false;
      return $_type.allocatedRangesOverlap(other);
    }
  }];
}

#endif // ASTER_AMDGCN_AMDGCNREGISTERTYPEINTERFACE_TD
