// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASTER_DIALECT_X86_IR_X86ENUMS_TD
#define ASTER_DIALECT_X86_IR_X86ENUMS_TD

include "mlir/IR/EnumAttr.td"

// x86_64 general-purpose registers, numbered 0..15 in Intel's REX-extended
// encoding (matches the ModR/M reg field for 64-bit operands).
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
// AMX tile registers (tmm0..tmm7).
//===----------------------------------------------------------------------===//

def X86_TMM0 : I32EnumAttrCase<"tmm0", 0>;
def X86_TMM1 : I32EnumAttrCase<"tmm1", 1>;
def X86_TMM2 : I32EnumAttrCase<"tmm2", 2>;
def X86_TMM3 : I32EnumAttrCase<"tmm3", 3>;
def X86_TMM4 : I32EnumAttrCase<"tmm4", 4>;
def X86_TMM5 : I32EnumAttrCase<"tmm5", 5>;
def X86_TMM6 : I32EnumAttrCase<"tmm6", 6>;
def X86_TMM7 : I32EnumAttrCase<"tmm7", 7>;

def X86_TmmEnum : I32EnumAttr<"TmmEnum",
    "AMX tile register name",
    [X86_TMM0, X86_TMM1, X86_TMM2, X86_TMM3,
     X86_TMM4, X86_TMM5, X86_TMM6, X86_TMM7]> {
  let cppNamespace = "::mlir::aster::x86";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ISA version marker.
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
