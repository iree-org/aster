//===- MemoryOpInterfaces.td - Memory Op Interfaces ----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN load and store operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_INTERFACES_MEMORYOPINTERFACES_TD
#define ASTER_DIALECT_AMDGCN_IR_INTERFACES_MEMORYOPINTERFACES_TD

include "mlir/IR/OpBase.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/AMDGCNInstOpInterface.td"
include "aster/Interfaces/DependentOpInterface.td"

def LoadOpInterface :
    OpInterface<"LoadOpInterface", [AMDGCNInstOpInterface, FutureOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for load operations providing uniform access to operands and
    results.
  }];
  let methods = [
    InterfaceMethod<"Get the destination operand.",
      "::mlir::aster::Operand", "getDst", (ins), [{}], [{
        return $_op.getDestMutable();
      }]>,
    InterfaceMethod<"Get the address operand.",
      "::mlir::aster::Operand", "getAddress", (ins), [{}], [{
        return $_op.getAddrMutable();
      }]>
  ];
  let extraTraitClassDeclaration = [{
    ::mlir::aster::Future getFuture() {
      return ::mlir::aster::Future::read($_op.getToken(), $_op.getDst().getValue());
    }
  }];
}

def StoreOpInterface :
    OpInterface<"StoreOpInterface", [AMDGCNInstOpInterface, FutureOpInterface]> {
  let cppNamespace = "::mlir::aster::amdgcn";
  let description = [{
    Interface for store operations providing uniform access to operands and
    results.
  }];
  let methods = [
    InterfaceMethod<"Get the data operand.",
      "::mlir::aster::Operand", "getDataOperand", (ins), [{}], [{
        return $_op.getDataMutable();
      }]>,
    InterfaceMethod<"Get the address operand.",
      "::mlir::aster::Operand", "getAddress", (ins), [{}], [{
        return $_op.getAddrMutable();
      }]>,
    InterfaceMethod<"Get the dynamic offset operand.",
      "::mlir::aster::Operand", "getDynamicOff", (ins), [{}], [{
        ::mlir::MutableOperandRange r = $_op.getDynamicOffsetMutable();
        return !r.empty() ? ::mlir::aster::Operand(r[0]) : ::mlir::aster::Operand();
      }]>
  ];
  let extraTraitClassDeclaration = [{
    ::mlir::aster::Future getFuture() {
      return ::mlir::aster::Future::write($_op.getToken(), $_op.getDataOperand().getValue());
    }
  }];
}

#endif // ASTER_DIALECT_AMDGCN_IR_INTERFACES_MEMORYOPINTERFACES_TD
