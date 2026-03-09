//===- AMDGCNAttrs.td - AMDGCN attrs ------------------------*- tablegen-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_ATTRS_TD
#define ASTER_AMDGCN_ATTRS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/KernelArgInterface.td"
include "aster/Dialect/AMDGCN/IR/Interfaces/HazardAttrInterface.td"
include "aster/Dialect/AMDGCN/IR/Hazards.td"
include "aster/Dialect/NormalForm/IR/NormalFormInterfaces.td"
include "aster/Interfaces/MemorySpaceConstraints.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

def AddressSpaceAttr : AMDGCN_Attr<"AddressSpace", "addr_space", [
    DeclareAttrInterfaceMethods<MemorySpaceAttrInterface>,
    DeclareAttrInterfaceMethods<MemorySpaceConstraintsAttrInterface>
  ]> {
  let summary = "Address space attribute";
  let parameters = (ins "AddressSpaceKind":$space, "AccessKind":$kind);
  let assemblyFormat = "`<` $space `,` $kind `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Inst attribute
//===----------------------------------------------------------------------===//

/// Attribute constraint for AMDGCN's InstAttr attribute.
def AMDGCN_InstAttr :
  Attr<CPred<"isa<::mlir::aster::amdgcn::InstAttr>($_self)">,
    "AMDGCN instruction attribute"> {
  let storageType = "::mlir::aster::amdgcn::InstAttr";
  let constBuilderCall =
      "::mlir::aster::amdgcn::InstAttr::get($_builder.getContext(), $0)";
  let returnType = "::mlir::aster::amdgcn::OpCode";
  let convertFromStorage = "$_self.getValue()";
}

//===----------------------------------------------------------------------===//
// Kernel argument attributes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ByValue argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN by-value argument attribute.
def ByValueArgAttr : AMDGCN_Attr<"ByValueArg", "by_val_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel by-value argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel by-value argument.

    Example:

    ```mlir
    #amdgcn.by_val_arg<size = 4, name = "int_arg", type = i32>,
    ```
  }];
  let parameters = (ins "uint32_t":$size,
                        OptionalParameter<"uint32_t">:$alignment,
                        StringRefParameter<"name", "\"\"">:$name,
                        OptionalParameter<"Type">:$type);
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      return ArgumentValueKind::ByValue;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Buffer argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN buffer argument attribute.
def BufferArgAttr : AMDGCN_Attr<"BufferArg", "buffer_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel buffer argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel buffer argument.

    Example:

    ```mlir
    #amdgcn.buffer_arg<address_space = generic, access = read_write, type = !ptr.ptr<#ptr.generic_space>>
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"AddressSpaceKind", "AddressSpaceKind::Global">:$address_space,
    DefaultValuedParameter<"AccessKind", "AccessKind::ReadWrite">:$access,
    DefaultValuedParameter<"KernelArgumentFlags", "KernelArgumentFlags::None">:$flags,
    StringRefParameter<"name", "\"\"">:$name,
    OptionalParameter<"Type">:$type
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      return ArgumentValueKind::GlobalBuffer;
    }
    // Get the actual memory access pattern.
    AccessKind getActualAccess() const {
      return getAccess();
    }
    // Get the alignment of the argument in bytes.
    std::optional<uint32_t> getAlignment() const {
      return 8u;
    }
    // Get the size of the argument in bytes.
    uint32_t getSize() const {
      return 8u;
    }
  }];
}

//===----------------------------------------------------------------------===//
// BlockDim argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN block dimension argument attribute.
def BlockDimArgAttr : AMDGCN_Attr<"BlockDimArg", "block_dim_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel block dimension argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel block dimension argument.

    Example:

    ```mlir
    #amdgcn.block_dim_arg<x>
    ```
  }];
  let parameters = (ins "Dim":$dimension);
  let assemblyFormat = "`<` $dimension `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      switch(getDimension()) {
        case Dim::X: return ArgumentValueKind::HiddenGroupSizeX;
        case Dim::Y: return ArgumentValueKind::HiddenGroupSizeY;
        case Dim::Z: return ArgumentValueKind::HiddenGroupSizeZ;
      }
    }
    // Get the alignment of the argument in bytes.
    std::optional<uint32_t> getAlignment() const {
      return 4u;
    }
    // Get the size of the argument in bytes.
    uint32_t getSize() const {
      return 4u;
    }
  }];
}

//===----------------------------------------------------------------------===//
// GridDim argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN grid dimension argument attribute.
def GridDimArgAttr : AMDGCN_Attr<"GridDimArg", "grid_dim_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel grid dimension argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel grid dimension argument.

    Example:

    ```mlir
    #amdgcn.grid_dim_arg<x>
    ```
  }];
  let parameters = (ins "Dim":$dimension);
  let assemblyFormat = "`<` $dimension `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      switch(getDimension()) {
        case Dim::X: return ArgumentValueKind::HiddenBlockCountX;
        case Dim::Y: return ArgumentValueKind::HiddenBlockCountY;
        case Dim::Z: return ArgumentValueKind::HiddenBlockCountZ;
      }
    }
    // Get the alignment of the argument in bytes.
    std::optional<uint32_t> getAlignment() const {
      return 4u;
    }
    // Get the size of the argument in bytes.
    uint32_t getSize() const {
      return 4u;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Kernel arguments attribute
//===----------------------------------------------------------------------===//

def KernelArgumentsAttr
  : ArrayOfAttr<AMDGCN_Dialect, "KernelArguments", "kernel_arguments",
                "::mlir::aster::amdgcn::KernelArgAttrInterface"> {
  let summary = "AMDGCN kernel arguments attribute";
  let constBuilderCall = [{
    $_builder.getAttr<KernelArgumentsAttr>(ArrayRef<KernelArgAttrInterface>($0))
  }];
  let defaultValue = "{}";
  let assemblyFormat = [{
    `<`(`[` $value^ `]`)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// Normal form attributes
//===----------------------------------------------------------------------===//

def NoValueSemanticRegistersAttr : AMDGCN_Attr<
    "NoValueSemanticRegisters", "no_value_semantic_registers",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyType"]>]> {
  let summary = "Normal form: no register types with value semantics";
  let description = [{
    Verifies that no register types in the IR have value semantics.
    This is the post-condition of the `amdgcn-to-register-semantics` pass,
    which converts all value-semantic allocas to unallocated register semantics.

    After this normal form holds, every `RegisterTypeInterface` type in the
    module has either `Unallocated` or `Allocated` semantics -- never `Value`.
  }];
}

def AllRegistersAllocatedAttr : AMDGCN_Attr<
    "AllRegistersAllocated", "all_registers_allocated",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyType"]>]> {
  let summary = "Normal form: all register types have allocated semantics";
  let description = [{
    Verifies that every register type in the IR has allocated semantics
    (i.e., has been assigned a concrete physical register).
    This is the post-condition of the `amdgcn-register-coloring` pass.

    After this normal form holds, no `RegisterTypeInterface` type has
    `Value` or `Unallocated` semantics -- all are `Allocated`.
  }];
}

def NoRegCastOpsAttr : AMDGCN_Attr<
    "NoRegCastOps", "no_reg_cast_ops",
    [DeclareAttrInterfaceMethods<NormalFormAttrInterface, ["verifyOperation"]>]> {
  let summary = "Normal form: no lsir.reg_cast operations";
  let description = [{
    Verifies that no `lsir.reg_cast` operations exist in the IR.
    This is the post-condition of the `aster-to-amdgcn` pass.

    A surviving `lsir.reg_cast` indicates an incorrect `lsir.to_reg` or
    `lsir.from_reg` remaining from above, potentially from high-level authored
    IR.
  }];
}

#endif // ASTER_AMDGCN_ATTRS_TD
