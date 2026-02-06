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
include "aster/Dialect/AMDGCN/IR/AMDGCNAttrInterfaces.td"
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

/// AMDGCN instruction attribute.
def AMDGCN_InstAttr : AMDGCN_Attr<"Inst", "inst"> {
  let summary = "AMDGCN instruction attribute";
  let description = [{
    Attribute representing an AMDGCN instruction opcode.

    Example:

    ```mlir
    #amdgcn.inst<v_move_b32>
    ```
  }];
  let parameters = (ins "mlir::aster::amdgcn::OpCode":$value);
  let constBuilderCall =
      cppNamespace#"::"#cppClassName#"::get($_builder.getContext(), $0)";
  let returnType = cppNamespace#"::OpCode";
  let convertFromStorage = "$_self.getValue()";
  let genStorageClass = 0;
  let assemblyFormat = [{ `<` $value `>` }];
  let extraClassDeclaration = [{
    const mlir::aster::amdgcn::InstMetadata* getMetadata() const;
  }];
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
// OpaqueModifier attribute
//===----------------------------------------------------------------------===//

/// AMDGCN opaque modifier attribute.
def OpaqueModifierAttr : AMDGCN_Attr<"OpaqueModifier", "opaque_modifier"> {
  let summary = "AMDGCN opaque modifier attribute";
  let description = [{
    Attribute representing an opaque modifier with a string value.

    Example:

    ```mlir
    #amdgcn.opaque_modifier<"neg">
    ```
  }];
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ModifiersAttr attribute
//===----------------------------------------------------------------------===//

def ModifiersAttr
  : ArrayOfAttr<AMDGCN_Dialect, "Modifiers", "modifiers",
                "::mlir::aster::amdgcn::OpaqueModifierAttr"> {
  let summary = "AMDGCN modifiers list attribute";
  let description = [{
    Attribute representing a list of opaque modifiers.

    Example:

    ```mlir
    #amdgcn.modifiers<[#amdgcn.opaque_modifier<"neg">, #amdgcn.opaque_modifier<"abs">]>
    ```
  }];
  let constBuilderCall = [{
    $_builder.getAttr<ModifiersAttr>(ArrayRef<OpaqueModifierAttr>($0))
  }];
  let defaultValue = "{}";
  let assemblyFormat = [{
    `<`(`[` $value^ `]`)? `>`
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

#endif // ASTER_AMDGCN_ATTRS_TD
