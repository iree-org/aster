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

class BuiltinAttrBuilder<Attr namedAttr, dag params> : PyAttr {
  let builder = OpBuilder<params, [{
    return $_ir.AttrBuilder.get('}] # !cast<string>(namedAttr) # [{')($_args, $_ctx)
  }]>;
}

// Instruction modifier for modifiers of the form:
//  ($name`(` $value `)` )?
// E.g., vmcnt(0)
// If the value is the default value, the modifier is omitted.
class ParenIntModifier<Attr intAttr, string name, int defaultValue, Attr intAttrBase = intAttr>
    : DefaultValuedAttr<intAttr, !cast<string>(defaultValue)>, AsmArgFormat,
      BuiltinAttrBuilder<intAttrBase, (ins "int":$value)>{
  let asmParser = "$_parser.parseParenIntModifier(\"" # name # "\", $_self);";
  let asmPrinter = "$_printer.printParenIntModifier(\"" # name # "\", $_self, " # defaultValue # ");";
}

// Instruction modifier for modifiers of the form:
//  ($name `:` `[` $value `]` )?
// E.g., cbsz:[4]
// If the value is the default value, the modifier is omitted.
class SquareIntModifier<Attr intAttr, string name, int defaultValue>
    : DefaultValuedAttr<intAttr, !cast<string>(defaultValue)>, AsmArgFormat,
      BuiltinAttrBuilder<intAttr, (ins "int":$value)> {
  let asmParser = "$_parser.parseSquareIntModifier(\"" # name # "\", $_self);";
  let asmPrinter = "$_printer.printSquareIntModifier(\"" # name # "\", $_self, " # defaultValue # ");";
}

// Instruction modifier for modifiers of the form:
//  ($name `:` $value )?
// E.g., offset:4
// If the value is the default value, the modifier is omitted.
class NamedIntModifier<Attr intAttr, string name, int defaultValue>
    : DefaultValuedAttr<intAttr, !cast<string>(defaultValue)>, AsmArgFormat,
      BuiltinAttrBuilder<intAttr, (ins "int":$value)> {
  let asmParser = "$_parser.parseIntModifier(\"" # name # "\", $_self);";
  let asmPrinter = "$_printer.printIntModifier(\"" # name # "\", $_self, " # defaultValue # ");";
}


// Instruction modifier for modifiers of the form:
//  ($value)?
// E.g., 4
// If the value is the default value, the modifier is omitted.
class IntModifier<Attr intAttr, int defaultValue, Attr intAttrBase = intAttr>
    : DefaultValuedAttr<intAttr, !cast<string>(defaultValue)>, AsmArgFormat,
      BuiltinAttrBuilder<intAttrBase, (ins "int":$value)> {
  let asmParser = "$_parser.parseIntModifier($_self);";
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

// IMM16 attribute
def Imm16Attr : IntModifier<I16Attr, 0, I16Attr> {
  let summary = "16 bit immediate value";
}

// Offset 16 attribute
def Offset16Attr : NamedIntModifier<I16Attr, "offset", 0> {
  let summary = "16 bit address offset";
}

// Offset 32 attribute
def Offset32Attr : NamedIntModifier<I32Attr, "offset", 0> {
  let summary = "32 bit address offset";
}
// TODO: Remove this case once SMem ops are correct.
def OffsetImm32Attr : IntModifier<I32Attr, 0> {
  let summary = "32 bit address imm offset";
  let asmPrinter = "$_printer.printIntModifier($_self, -1);";
}

// EXPCNT attribute
def ExpcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<8>]>, "expcnt", 8, I8Attr> {
  let summary = "EXPCNT wait count modifier";
}

// LGKMCNT attribute
def LgkmcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<16>]>, "lgkmcnt", 16, I8Attr> {
  let summary = "LGKMCNT wait count modifier";
}

// VMCNT attribute
def VmcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<64>]>, "vmcnt", 64, I8Attr> {
  let summary = "VMCNT wait count modifier";
}

#endif // ASTER_AMDGCN_AMDGCNMODIFIERS_TD
