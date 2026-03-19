//===- LayoutDialect.td - Layout dialect -----------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LAYOUT_IR_LAYOUTDIALECT_TD
#define ASTER_DIALECT_LAYOUT_IR_LAYOUTDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Layout dialect definition
//===----------------------------------------------------------------------===//

def Layout_Dialect : Dialect {
  let name = "layout";
  let summary = "Layout algebra dialect for composable data layout manipulation";
  let description = [{
    The Layout dialect provides operations for composable data layout
    manipulation. A layout maps logical coordinates to physical offsets.

    The dialect is designed to support and experiment with different types of
    layouts in a composable fashion. Over time, this may include:
    - Strided memref layouts (the standard MLIR representation)
    - Triton F2 linear layouts
    - CuTe hierarchical layouts
    - Future layouts with sparsity, compression, or non-linear mappings

    The simplest layout is a flat shape:stride pair -- an explicit
    representation of the strided layout already used by upstream MLIR memrefs.
  }];
  let cppNamespace = "::mlir::aster::layout";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    private:
      void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class Layout_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Layout_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class Layout_Op<string mnemonic, list<Trait> traits = []>
    : Op<Layout_Dialect, mnemonic, traits>;

#endif // ASTER_DIALECT_LAYOUT_IR_LAYOUTDIALECT_TD
