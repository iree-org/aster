//===-- AMDGCNAttrInterfaces.td - AMDGCN attr interfaces ------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_ATTRINTERFACES
#define AMDGCN_ATTRINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"

def KernelArgInterface : AttrInterface<"KernelArgAttrInterface"> {
  let description = [{
      Interface for kernel argument attributes that provide metadata
      about kernel parameters including layout, access patterns, and qualifiers.
  }];
  let cppNamespace = "::mlir::aster::amdgcn";
  let methods = [
    InterfaceMethod<
      "Get kernel argument name",
      "::llvm::StringRef", "getName", (ins), [{}], [{
        return "";
      }]
    >,
    InterfaceMethod<
      "Get kernel argument type",
      "::mlir::Type", "getType", (ins), [{}], [{
        return {};
      }]
    >,
    InterfaceMethod<
      "Get kernel argument size in bytes",
      "uint32_t", "getSize", (ins)
    >,
    InterfaceMethod<
      "Get kernel argument alignment in bytes",
      "std::optional<uint32_t>", "getAlignment", (ins), [{}], [{
        return std::nullopt;
      }]
    >,
    InterfaceMethod<
      "Get kernel argument value kind",
      "::mlir::aster::amdgcn::ArgumentValueKind", "getValueKind", (ins)
    >,
    InterfaceMethod<
      "Get pointee alignment for pointer arguments",
      "::std::optional<uint32_t>", "getPointeeAlign", (ins), [{}], [{
        return std::nullopt;
      }]
    >,
    InterfaceMethod<
      "Get kernel argument address space",
      "::mlir::aster::amdgcn::AddressSpaceKind", "getAddressSpace", (ins), [{}], [{
        return ::mlir::aster::amdgcn::AddressSpaceKind::Invalid;
      }]
    >,
    InterfaceMethod<
      "Get kernel argument access qualifier",
      "::mlir::aster::amdgcn::AccessKind", "getAccess", (ins), [{}], [{
        return ::mlir::aster::amdgcn::AccessKind::Unspecified;
      }]
    >,
    InterfaceMethod<
      "Get actual memory access pattern",
      "::mlir::aster::amdgcn::AccessKind", "getActualAccess", (ins), [{}], [{
        return ::mlir::aster::amdgcn::AccessKind::Unspecified;
      }]
    >,
    InterfaceMethod<
      "Get kernel argument flags",
      "::mlir::aster::amdgcn::KernelArgumentFlags", "getFlags", (ins), [{}], [{
        return ::mlir::aster::amdgcn::KernelArgumentFlags::None;
      }]
    >
  ];
}

#endif // AMDGCN_ATTRINTERFACES
