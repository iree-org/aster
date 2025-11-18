//===- ModuleOpInterface.td - Module operation interface ---*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ModuleOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_MODULEOPINTERFACE_TD
#define ASTER_INTERFACES_MODULEOPINTERFACE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ModuleOpInterface
//===----------------------------------------------------------------------===//

def ModuleOpInterface : OpInterface<"ModuleOpInterface"> {
  let cppNamespace = "::mlir::aster";

  let description = [{
    This interface identifies operations that represent modules or compilation
    units. Operations implementing this interface can contain functions,
    global variables, and other top-level declarations.

    This interface is used by passes and analyses that need to identify
    module boundaries.
  }];
}

#endif // ASTER_INTERFACES_MODULEOPINTERFACE_TD
