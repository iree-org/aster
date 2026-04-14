// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_AMX_IR_AMXTYPES_TD
#define ASTER_DIALECT_AMX_IR_AMXTYPES_TD

include "aster/Dialect/AMX/IR/AMXDialect.td"

def AMX_TileType : AMX_Type<"Tile", "tile"> {
  let summary = "Pre-assigned x86 AMX tile register";
  let description = [{
    An AMX tile bound to a physical `tmmN` register, with a logical
    shape and element type.

    Syntax:
    ```mlir
    !amx.tile<"tmm0", 16 x 16 x f32>
    !amx.tile<"tmm1", 16 x 32 x bf16>
    ```
  }];
  let parameters = (ins
    "::mlir::StringAttr":$name,
    "int64_t":$rows,
    "int64_t":$cols,
    "::mlir::Type":$elementType
  );
  let assemblyFormat =
    "`<` $name `,` $rows `x` $cols `x` $elementType `>`";
}

def AMX_AnyTile : Type<
  CPred<"::llvm::isa<::mlir::aster::amx::TileType>($_self)">,
  "amx tile">;

#endif // ASTER_DIALECT_AMX_IR_AMXTYPES_TD
