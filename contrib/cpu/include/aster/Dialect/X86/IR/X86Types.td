// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86TYPES_TD
#define ASTER_DIALECT_X86_IR_X86TYPES_TD

include "aster/Dialect/X86/IR/X86Dialect.td"
include "aster/Dialect/X86/IR/X86Enums.td"

def X86_GprType : X86_Type<"Gpr", "gpr"> {
  let summary = "Pre-assigned x86_64 general-purpose register";
  let description = [{
    An x86_64 GPR bound to a specific physical register.

    ```mlir
    !x86.gpr<rdi>
    !x86.gpr<r8>
    ```
  }];
  let parameters = (ins EnumParameter<X86_GprEnum>:$reg);
  let assemblyFormat = "`<` $reg `>`";
}

def X86_AnyGpr : Type<
  CPred<"::llvm::isa<::mlir::aster::x86::GprType>($_self)">,
  "x86 gpr">;

//===----------------------------------------------------------------------===//
// AVX / AVX2 / AVX512 vector register types.
// Register index is a bare integer; the type name carries the class.
//===----------------------------------------------------------------------===//

def X86_XMMType : X86_Type<"XMM", "avx.xmm"> {
  let summary = "Pre-assigned AVX 128-bit vector register (xmm0..xmm15)";
  let description = [{
    ```mlir
    !x86.avx.xmm<0>
    ```
  }];
  let parameters = (ins "int64_t":$reg);
  let assemblyFormat = "`<` $reg `>`";
}

def X86_YMMType : X86_Type<"YMM", "avx2.ymm"> {
  let summary = "Pre-assigned AVX2 256-bit vector register (ymm0..ymm15)";
  let description = [{
    ```mlir
    !x86.avx2.ymm<0>
    ```
  }];
  let parameters = (ins "int64_t":$reg);
  let assemblyFormat = "`<` $reg `>`";
}

def X86_ZMMType : X86_Type<"ZMM", "avx512.zmm"> {
  let summary = "Pre-assigned AVX512 512-bit vector register (zmm0..zmm31)";
  let description = [{
    ```mlir
    !x86.avx512.zmm<0>
    ```
  }];
  let parameters = (ins "int64_t":$reg);
  let assemblyFormat = "`<` $reg `>`";
}

def X86_AnyXMM : Type<
  CPred<"::llvm::isa<::mlir::aster::x86::XMMType>($_self)">,
  "x86 avx xmm">;

def X86_AnyYMM : Type<
  CPred<"::llvm::isa<::mlir::aster::x86::YMMType>($_self)">,
  "x86 avx2 ymm">;

def X86_AnyZMM : Type<
  CPred<"::llvm::isa<::mlir::aster::x86::ZMMType>($_self)">,
  "x86 avx512 zmm">;

//===----------------------------------------------------------------------===//
// AMX tile register (tmm0..tmm7) with shape and element type.
//===----------------------------------------------------------------------===//

def X86_TMMType : X86_Type<"TMM", "amx.tmm"> {
  let summary = "Pre-assigned AMX tile register with shape (tmm0..tmm7)";
  let description = [{
    ```mlir
    !x86.amx.tmm<0, 16 x 16 x f32>
    !x86.amx.tmm<1, 16 x 32 x bf16>
    ```
  }];
  let parameters = (ins
    "int64_t":$reg,
    "int64_t":$rows,
    "int64_t":$cols,
    "::mlir::Type":$elementType
  );
  let assemblyFormat =
    "`<` $reg `,` $rows `x` $cols `x` $elementType `>`";
}

def X86_AnyTMM : Type<
  CPred<"::llvm::isa<::mlir::aster::x86::TMMType>($_self)">,
  "x86 amx tmm">;

#endif // ASTER_DIALECT_X86_IR_X86TYPES_TD
