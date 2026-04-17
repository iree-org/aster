// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86OPS_TD
#define ASTER_DIALECT_X86_IR_X86OPS_TD

include "aster/Dialect/AMX/IR/Interfaces/AMXAsmOpInterface.td"
include "aster/Dialect/X86/IR/Interfaces/X86IsaOpInterface.td"
include "aster/Dialect/X86/IR/X86Attrs.td"
include "aster/Dialect/X86/IR/X86Dialect.td"
include "aster/Dialect/X86/IR/X86Types.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class X86_Op<string mnemonic, list<Trait> traits = []>
    : Op<X86_Dialect, mnemonic, traits>;

// Base class for ops that emit one line of AT&T asm.
class X86_AsmOp<string mnemonic, list<Trait> traits = []>
    : X86_Op<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<AMXAsmOpInterface>])>;

//===----------------------------------------------------------------------===//
// AVX / AVX2 / AVX512 FMA ops.
//===----------------------------------------------------------------------===//

class X86_FmaOp<string mnemonic, string isa, Type regTy>
    : X86_AsmOp<mnemonic, [
        DeclareOpInterfaceMethods<X86IsaOpInterface>,
        AllTypesMatch<["acc", "result"]>]> {
  let arguments = (ins regTy:$acc, regTy:$lhs, regTy:$rhs);
  let results = (outs regTy:$result);
  let assemblyFormat = [{
    $acc `,` $lhs `,` $rhs attr-dict
      `:` functional-type(operands, results)
  }];
  string x86IsaVersion = isa;
}

def X86_AVXVFmaddOp : X86_FmaOp<"avx.vfmadd231ps", "avx", X86_AnyXMM> {
  let summary = "FMA on AVX 128-bit registers (acc += lhs * rhs)";
}

def X86_AVX2VFmaddOp : X86_FmaOp<"avx2.vfmadd231ps", "avx2", X86_AnyYMM> {
  let summary = "FMA on AVX2 256-bit registers (acc += lhs * rhs)";
}

def X86_AVX512VFmaddOp : X86_FmaOp<"avx512.vfmadd231ps", "avx512", X86_AnyZMM> {
  let summary = "FMA on AVX512 512-bit registers (acc += lhs * rhs)";
}

//===----------------------------------------------------------------------===//
// AVX / AVX2 / AVX512 load/store ops.
//===----------------------------------------------------------------------===//

class X86_VecLoadOp<string mnemonic, string isa, Type regTy>
    : X86_AsmOp<mnemonic, [DeclareOpInterfaceMethods<X86IsaOpInterface>]> {
  let arguments = (ins StrAttr:$inst, X86_AnyGpr:$base);
  let results = (outs regTy:$result);
  let assemblyFormat = [{
    custom<BareKeyword>($inst) $base attr-dict `:` type($base) `->` type($result)
  }];
  string x86IsaVersion = isa;
}

class X86_VecStoreOp<string mnemonic, string isa, Type regTy>
    : X86_AsmOp<mnemonic, [DeclareOpInterfaceMethods<X86IsaOpInterface>]> {
  let arguments = (ins StrAttr:$inst, X86_AnyGpr:$base, regTy:$value);
  let assemblyFormat = [{
    custom<BareKeyword>($inst) $base `,` $value attr-dict `:` type($base) `,` type($value)
  }];
  string x86IsaVersion = isa;
}

def X86_AVXLoadOp    : X86_VecLoadOp<"avx.load",    "avx",    X86_AnyXMM>;
def X86_AVX2LoadOp   : X86_VecLoadOp<"avx2.load",   "avx2",   X86_AnyYMM>;
def X86_AVX512LoadOp : X86_VecLoadOp<"avx512.load", "avx512", X86_AnyZMM>;

def X86_AVXStoreOp    : X86_VecStoreOp<"avx.store",    "avx",    X86_AnyXMM>;
def X86_AVX2StoreOp   : X86_VecStoreOp<"avx2.store",   "avx2",   X86_AnyYMM>;
def X86_AVX512StoreOp : X86_VecStoreOp<"avx512.store", "avx512", X86_AnyZMM>;

//===----------------------------------------------------------------------===//
// X86 Module Op for ISA-level gating.
//===----------------------------------------------------------------------===//

def X86_ModuleOp : X86_Op<"module", [
    IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    SingleBlock, NoTerminator
  ]> {
  let summary = "x86 module with ISA target level";
  let description = [{
    A container for x86 functions. The `target_isa` attribute declares the
    ISA target (avx / avx2 / avx512). A region verifier ensures every
    `X86IsaOpInterface` op in the body has `getRequiredIsa() <= target_isa`.

    Example:
    ```mlir
    x86.module @my_kernel target_isa = #x86.isa<avx512> {
      func.func @matmul(...) {
        // avx, avx2, and avx512 ops all allowed here.
      }
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    X86_IsaVersionAttr:$target_isa
  );
  let regions = (region SizedRegion<1>:$body_region);
  let assemblyFormat = [{
    $sym_name `target_isa` `=` $target_isa attr-dict-with-keyword $body_region
  }];
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
    bool isOptionalSymbol() { return false; }
  }];
}

#endif // ASTER_DIALECT_X86_IR_X86OPS_TD
