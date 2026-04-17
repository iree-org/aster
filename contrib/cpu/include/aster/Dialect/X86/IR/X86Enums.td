// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86ENUMS_TD
#define ASTER_DIALECT_X86_IR_X86ENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// x86_64 general-purpose registers (Intel REX encoding 0..15).
//===----------------------------------------------------------------------===//

def X86_RAX : I32EnumAttrCase<"rax",  0>;
def X86_RBX : I32EnumAttrCase<"rbx",  1>;
def X86_RCX : I32EnumAttrCase<"rcx",  2>;
def X86_RDX : I32EnumAttrCase<"rdx",  3>;
def X86_RBP : I32EnumAttrCase<"rbp",  4>;
def X86_RDI : I32EnumAttrCase<"rdi",  5>;
def X86_RSP : I32EnumAttrCase<"rsp",  6>;
def X86_RSI : I32EnumAttrCase<"rsi",  7>;
def X86_R8  : I32EnumAttrCase<"r8",   8>;
def X86_R9  : I32EnumAttrCase<"r9",   9>;
def X86_R10 : I32EnumAttrCase<"r10", 10>;
def X86_R11 : I32EnumAttrCase<"r11", 11>;
def X86_R12 : I32EnumAttrCase<"r12", 12>;
def X86_R13 : I32EnumAttrCase<"r13", 13>;
def X86_R14 : I32EnumAttrCase<"r14", 14>;
def X86_R15 : I32EnumAttrCase<"r15", 15>;

def X86_GprEnum : I32EnumAttr<"GprEnum",
    "x86_64 general-purpose register name (Intel REX encoding 0..15)",
    [X86_RAX, X86_RCX, X86_RDX, X86_RBX, X86_RSP, X86_RBP, X86_RSI, X86_RDI,
     X86_R8,  X86_R9,  X86_R10, X86_R11, X86_R12, X86_R13, X86_R14, X86_R15]> {
  let cppNamespace = "::mlir::aster::x86";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ISA version marker.
// Each op declares the minimum required ISA level, and a function-level
// target marker `{x86.isa = #x86.isa<avx512>}` lets a verifier check
// that every op in the body is available on the chosen target.
//
// Assumes a newer target can always execute every op from older target;
//   i.e. `avx512 > avx2 > avx`.
//===----------------------------------------------------------------------===//

def X86_IsaAVX    : I32EnumAttrCase<"avx",    0>;
def X86_IsaAVX2   : I32EnumAttrCase<"avx2",   1>;
def X86_IsaAVX512 : I32EnumAttrCase<"avx512", 2>;

def X86_IsaVersionEnum : I32EnumAttr<"IsaVersion",
    "x86_64 SIMD ISA extension level",
    [X86_IsaAVX, X86_IsaAVX2, X86_IsaAVX512]> {
  let cppNamespace = "::mlir::aster::x86";
  let genSpecializedAttr = 0;
}

#endif // ASTER_DIALECT_X86_IR_X86ENUMS_TD
