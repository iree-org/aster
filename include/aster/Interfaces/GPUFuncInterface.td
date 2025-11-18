//===- GPUFuncInterface.td - GPU func op interface ---------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_GPUFUNCINTERFACE_TD
#define ASTER_INTERFACES_GPUFUNCINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPUFunc Interface
//===----------------------------------------------------------------------===//

def GPUFuncInterface : OpInterface<"GPUFuncInterface"> {
  let description = [{
    Interface for gpu function operations.
  }];
  let cppNamespace = "::mlir::aster";
  let methods = [
    InterfaceMethod<
      "Get the grid dimensions",
      "::mlir::ArrayRef<int32_t>", "getGridDims"
    >,
    InterfaceMethod<
      "Get the block dimensions",
      "::mlir::ArrayRef<int32_t>", "getBlockDims"
    >,
    InterfaceMethod<
      "Set the grid dimensions",
      "void", "setGridDims", (ins "::mlir::ArrayRef<int32_t>":$dims)
    >,
    InterfaceMethod<
      "Set the block dimensions",
      "void", "setBlockDims", (ins "::mlir::ArrayRef<int32_t>":$dims)
    >,
    InterfaceMethod<
      "Return whether the function is a GPU kernel",
      "bool", "isGPUKernel"
    >,
    InterfaceMethod<
      "Mark the function as a GPU kernel",
      "void", "setGPUKernel", (ins "bool":$isKernel)
    >,
    InterfaceMethod<
      "Get the host ABI function type",
      "std::tuple<::mlir::FunctionType, ::mlir::ArrayRef<int32_t>, ::mlir::ArrayRef<int32_t>>", "getHostABI"
    >,
    InterfaceMethod<
      "Set the host ABI function type",
      "void", "setHostABI", (ins "::mlir::FunctionType":$type, "::mlir::ArrayRef<int32_t>":$sizeInBytes, "::mlir::ArrayRef<int32_t>":$alignInBytes)
    >,
    InterfaceMethod<
      "Get the amount of shared memory in bytes",
      "int32_t", "getSharedMemorySize"
    >,
    InterfaceMethod<
      "Set the amount of shared memory in bytes",
      "void", "setSharedMemorySize", (ins "int32_t":$sizeInBytes)
    >,
  ];
}

#endif // ASTER_INTERFACES_GPUFUNCINTERFACE_TD
