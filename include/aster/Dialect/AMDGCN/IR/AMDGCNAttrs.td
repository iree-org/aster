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
// This file defines the amdgcn dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_ATTRS
#define AMDGCN_ATTRS

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNAttrInterfaces.td"

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

//===----------------------------------------------------------------------===//
// Kernel argument attributes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ByValue argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN by-value argument attribute.
def ByValueArgAttr : AMDGCN_Attr<"ByValueArg", "by_val_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel by-value argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel by-value argument.

    Example:

    ```mlir
    #amdgcn.by_val_arg<size = 4, name = "int_arg", type = i32>,
    ```
  }];
  let parameters = (ins "uint32_t":$size,
                        OptionalParameter<"uint32_t">:$alignment,
                        StringRefParameter<"name", "\"\"">:$name,
                        OptionalParameter<"Type">:$type);
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      return ArgumentValueKind::ByValue;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Buffer argument attribute
//===----------------------------------------------------------------------===//

/// AMDGCN buffer argument attribute.
def BufferArgAttr : AMDGCN_Attr<"BufferArg", "buffer_arg", [
    KernelArgInterface
  ]> {
  let summary = "AMDGCN kernel buffer argument attribute";
  let description = [{
    Attribute representing an AMDGCN kernel buffer argument.

    Example:

    ```mlir
    #amdgcn.buffer_arg<address_space = generic, access = read_write, type = !ptr.ptr<#ptr.generic_space>>
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"AddressSpaceKind", "AddressSpaceKind::Global">:$address_space,
    DefaultValuedParameter<"AccessKind", "AccessKind::ReadWrite">:$access,
    DefaultValuedParameter<"KernelArgumentFlags", "KernelArgumentFlags::None">:$flags,
    StringRefParameter<"name", "\"\"">:$name,
    OptionalParameter<"Type">:$type
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // KernelArgInterface methods
    //===------------------------------------------------------------------===//
    /// Get the value kind of the kernel argument.
    ArgumentValueKind getValueKind() const {
      return ArgumentValueKind::GlobalBuffer;
    }
    // Get the actual memory access pattern.
    AccessKind getActualAccess() const {
      return getAccess();
    }
    // Get the alignment of the argument in bytes.
    std::optional<uint32_t> getAlignment() const {
      return 8u;
    }
    // Get the size of the argument in bytes.
    uint32_t getSize() const {
      return 8u;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Kernel arguments attribute
//===----------------------------------------------------------------------===//

def KernelArgumentsAttr
  : ArrayOfAttr<AMDGCN_Dialect, "KernelArguments", "kernel_arguments",
                "::mlir::aster::amdgcn::KernelArgAttrInterface"> {
  let summary = "AMDGCN kernel arguments attribute";
  let constBuilderCall = [{
    $_builder.getAttr<KernelArgumentsAttr>(ArrayRef<KernelArgAttrInterface>($0))
  }];
  let defaultValue = "{}";
  let assemblyFormat = [{
    `<`(`[` $value^ `]`)? `>`
  }];
}

//===----------------------------------------------------------------------===//
// Enum attributes
//===----------------------------------------------------------------------===//

def RegisterKindAttr : AMDGCN_EnumAttr<"RegisterKindAttr", RegisterKind> {
  let summary = "AMDGCN register kind attribute";
}

def WorkitemIDModeAttr : AMDGCN_EnumAttr<"WorkitemIDModeAttr", WorkitemIDMode> {
  let summary = "AMDGCN VGPR workitem ID mode attribute";
  let defaultValue = "WorkitemIDMode::X";
}

def FloatRoundModeAttr : AMDGCN_EnumAttr<"FloatRoundModeAttr", FloatRoundMode> {
  let summary = "AMDGCN floating point rounding mode attribute";
  let defaultValue = "FloatRoundMode::NearEven";
}

def FloatDenormModeAttr : AMDGCN_EnumAttr<"FloatDenormModeAttr", FloatDenormMode> {
  let summary = "AMDGCN floating point denorm mode attribute";
  let defaultValue = "FloatDenormMode::SrcDst";
}

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
def ExpcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<7>]>, "expcnt", 7, I8Attr> {
  let summary = "EXPCNT wait count modifier";
}

// LGKMCNT attribute
def LgkmcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<15>]>, "lgkmcnt", 15, I8Attr> {
  let summary = "LGKMCNT wait count modifier";
}

// VMCNT attribute
def VmcntAttr : ParenIntModifier<ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<63>]>, "vmcnt", 63, I8Attr> {
  let summary = "VMCNT wait count modifier";
}

#endif // AMDGCN_ATTRS
