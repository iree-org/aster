//===- AMDGCNAttrs.td - AMDGCN attrs ------------------------*- tablegen-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN dialect modifier attributes.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_AMDGCNMODIFIERS_TD
#define ASTER_AMDGCN_AMDGCNMODIFIERS_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"

//===----------------------------------------------------------------------===//
// Inst modifier helpers
//===----------------------------------------------------------------------===//

// Instruction modifier for modifiers of the form:
//  ($name`(` $value `)` )?
// E.g., vmcnt(0)
// If the value is the default value, the modifier is omitted.
class ParenIntModifier<Attr intAttr, string name, int defaultValue>
    : DefaultValuedAttr<intAttr, !cast<string>(defaultValue)>, ASMArgFormat{
  let asmPrinter = "$_printer.printParenIntModifier(\"" # name # "\", $_self, " # defaultValue # ");";
}

// Instruction modifier for modifiers of the form:
//  ($name `:` `[` $value `]` )?
// E.g., cbsz:[4]
// If the value is the default value, the modifier is omitted.
class SquareIntModifier<Attr intAttr, string name, int defaultValue>
    : DefaultValuedAttr<intAttr, !cast<string>(defaultValue)>, ASMArgFormat {
  let asmPrinter = "$_printer.printSquareIntModifier(\"" # name # "\", $_self, " # defaultValue # ");";
}

// Instruction modifier for modifiers of the form:
//  ($name `:` $value )?
// E.g., offset:4
// If the value is the default value, the modifier is omitted.
class NamedIntModifier<Attr intAttr, string name, int defaultValue>
    : DefaultValuedAttr<intAttr, !cast<string>(defaultValue)>, ASMArgFormat {
  let asmPrinter = "$_printer.printIntModifier(\"" # name # "\", $_self, " # defaultValue # ");";
}


// Instruction modifier for modifiers of the form:
//  ($value)?
// E.g., 4
// If the value is the default value, the modifier is omitted.
class IntModifier<Attr intAttr, int defaultValue>
    : DefaultValuedAttr<intAttr, !cast<string>(defaultValue)>, ASMArgFormat {
  let asmPrinter = "$_printer.printIntModifier($_self, " # defaultValue # ");";
}

//===----------------------------------------------------------------------===//
// Inst modifiers
//===----------------------------------------------------------------------===//

// ABID attribute
def ABIDAttr : SquareIntModifier<I16Attr, "abid", 0> {
  let summary = "Matrix A group select id";
}

// BLGP attribute
def BLGPAttr : SquareIntModifier<I8Attr, "blgp", 0> {
  let summary = "Matrix B lane group pattern";
}

// CBSZ attribute
def CBSZAttr : SquareIntModifier<I8Attr, "cbsz", 0> {
  let summary = "Broadcast mode";
}

// Op-sel hi modifier base class.
// TODO: Revisit the default values.
class OpSelHiBase<int num_srcs> : ConfinedAttr<
    DefaultValuedAttr<DenseBoolArrayAttr, "{false, false, false}">, [DenseArrayCount<num_srcs>]
  >, ASMArgFormat {
  let summary = "op-sel high modifier";
  let asmPrinter = [{
    {
      $_printer.printKeyword("op_sel_hi");
      $_printer.getStream() << ":[";
      llvm::interleaveComma($_self, $_printer.getStream(), [&](bool value) {
        $_printer.getStream() << (int)value;
      });
      $_printer.getStream() << "] ";
    }
  }];
}

// WMMA neg modifier: a 3-element src-negate bit vector printed as
//   ($name `:` `[` a `,` b `,` c `]`)?
// In practice, neg_lo / neg_hi are only used on the C operand (src2 lane).
def NEGLOAttr : ConfinedAttr<
    DefaultValuedAttr<DenseBoolArrayAttr, "{false, false, false}">, [DenseArrayCount<3>]
  >, ASMArgFormat {
  let summary = "WMMA neg_lo:[a,b,c] source-negate modifier";
  let asmPrinter = [{
    if (!llvm::all_of($_self, [](bool v) { return !v; })) {
      $_printer.printKeyword("neg_lo");
      $_printer.getStream() << ":[";
      llvm::interleave($_self, $_printer.getStream(),
                       [&](bool value) { $_printer.getStream() << (int)value; }, ",");
      $_printer.getStream() << "] ";
    }
  }];
}

def NEGHIAttr : ConfinedAttr<
    DefaultValuedAttr<DenseBoolArrayAttr, "{false, false, false}">, [DenseArrayCount<3>]
  >, ASMArgFormat {
  let summary = "WMMA neg_hi:[a,b,c] source-negate modifier";
  let asmPrinter = [{
    if (!llvm::all_of($_self, [](bool v) { return !v; })) {
      $_printer.printKeyword("neg_hi");
      $_printer.getStream() << ":[";
      llvm::interleave($_self, $_printer.getStream(),
                       [&](bool value) { $_printer.getStream() << (int)value; }, ",");
      $_printer.getStream() << "] ";
    }
  }];
}

// IMM16 attribute -- always printed in ASM (SOPP instructions require the
// immediate operand even when it is zero, e.g. s_setprio 0).
def Imm16Attr : DefaultValuedAttr<I16Attr, "0">, ASMArgFormat {
  let summary = "16 bit immediate value";
  let asmPrinter = "$_printer.printIntModifierAlways($_self);";
}

// Signed i16 immediate for GFX12.5 barrier object ids (-1 = workgroup barrier).
def SignedImm16Attr : DefaultValuedAttr<I16Attr, "0">, ASMArgFormat {
  let summary = "Signed 16-bit immediate value";
  let asmPrinter = "$_printer.printImm16AsSigned($_self);";
}

// EXPCNT attribute
def ExpcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<8>]>, "expcnt", 8> {
  let summary = "EXPCNT wait count modifier";
}

// LGKMCNT attribute
def LgkmcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<16>]>, "lgkmcnt", 16> {
  let summary = "LGKMCNT wait count modifier";
}

// VMCNT attribute
def VmcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<64>]>, "vmcnt", 64> {
  let summary = "VMCNT wait count modifier";
}

#endif // ASTER_AMDGCN_AMDGCNMODIFIERS_TD
