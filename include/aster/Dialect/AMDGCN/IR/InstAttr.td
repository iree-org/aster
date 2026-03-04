//===- InstAttr.td - AMDGCN instruction attribute ----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN instruction attribute (InstAttr).
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_INSTATTR_TD
#define ASTER_AMDGCN_INSTATTR_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"

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

#endif // ASTER_AMDGCN_INSTATTR_TD
