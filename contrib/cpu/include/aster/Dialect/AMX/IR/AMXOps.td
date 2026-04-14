// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_AMX_IR_AMXOPS_TD
#define ASTER_DIALECT_AMX_IR_AMXOPS_TD

include "aster/Dialect/AMX/IR/AMXDialect.td"
include "aster/Dialect/AMX/IR/AMXTypes.td"
include "aster/Dialect/AMX/IR/Interfaces/AMXAsmOpInterface.td"
include "aster/Dialect/X86/IR/X86Types.td"
include "mlir/IR/OpBase.td"

class AMX_AsmOp<string mnemonic, list<Trait> traits = []>
    : AMX_Op<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<AMXAsmOpInterface>])>;

def AMX_LdTileCfgOp : AMX_AsmOp<"ldtilecfg"> {
  let summary = "Load AMX tile configuration (analyzed from function)";
  let description = [{
    Loads the TILECFG architectural register info derived from the tile types
    used in the enclosing function.

    The config is fully determined by static type information.
    The translation pass:
      1. walks the function, collects unique `!amx.tile<"tmmN", ...>` types,
      2. synthesizes the 64-byte config into a `.rodata` label
         `.Lcfg_<funcname>`, and
      3. emits `ldtilecfg .Lcfg_<funcname>(%rip)`.

    See `LDTILECFG` in Intel Architectures Software Developer's Manual.
    See llvm/lib/Target/X86/X86PreTileConfig.cpp for an implementation detail.
  }];
  let assemblyFormat = "attr-dict";
}

def AMX_TileLoaddOp : AMX_AsmOp<"tileloadd"> {
  let summary = "Load a tile from memory into tmm";
  let arguments = (ins X86_AnyGpr:$base, X86_AnyGpr:$stride);
  let results = (outs AMX_AnyTile:$result);
  let assemblyFormat = [{
    $base `,` $stride attr-dict
      `:` `(` type($base) `,` type($stride) `)` `->` type($result)
  }];
}

def AMX_TdpBf16PsOp : AMX_AsmOp<"tdpbf16ps",
    [AllTypesMatch<["acc", "result"]>]> {
  let summary = "BF16 tile dot-product accumulate";
  let description = [{
    `result = acc + lhs * rhs` on BF16 tiles with FP32 accumulation.
    The accumulator register is read-and-written in place: `acc` and `result`
    share the same physical tmm.
  }];
  let arguments = (ins
    AMX_AnyTile:$acc,
    AMX_AnyTile:$lhs,
    AMX_AnyTile:$rhs);
  let results = (outs AMX_AnyTile:$result);
  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs attr-dict
      `:` functional-type(operands, results)
  }];
}

def AMX_TileStoredOp : AMX_AsmOp<"tilestored"> {
  let summary = "Store a tile from tmm to memory";
  let arguments = (ins
    X86_AnyGpr:$base,
    X86_AnyGpr:$stride,
    AMX_AnyTile:$value);
  let assemblyFormat = [{
    $base `,` $stride `,` $value attr-dict
      `:` type($base) `,` type($stride) `,` type($value)
  }];
}

def AMX_TileReleaseOp : AMX_AsmOp<"tilerelease"> {
  let summary = "Release all AMX tile state";
  let assemblyFormat = "attr-dict";
}

#endif // ASTER_DIALECT_AMX_IR_AMXOPS_TD
