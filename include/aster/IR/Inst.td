//===- Inst.td - Instruction base definitions --------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines base classes and infrastructure for Aster instructions.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_INST_TD
#define ASTER_IR_INST_TD

include "aster/Interfaces/InstOpInterface.td"
include "aster/IR/Util.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Encodings and architectures
//===----------------------------------------------------------------------===//

// An architecture.
class Arch<string name> : EnumCaseInfo<name>;

// An instruction encoding.
class Encoding<string name> : PrettyEnumCaseInfo<name>;

// A concrete encoding of an instruction for a specific architecture.
class EncodedArch<Encoding _encoding, Arch _arch> {
  Arch arch = _arch;
  Encoding encoding = _encoding;
}

def pred;

// A concrete instruction encoding with opcode.
class InstEnc<list<EncodedArch> _archs, int _opcode, dag _constraints = (pred)> {
  list<EncodedArch> archs = _archs;
  int opcode = _opcode;
  dag constraints = _constraints;
}

//===----------------------------------------------------------------------===//
// Assembly syntax
//===----------------------------------------------------------------------===//

// Assembly syntax record carrying arch list, syntax string, and an optional predicate.
class ASMString<list<EncodedArch> _archs, string _syntax, Pred _pred = TruePred> {
  list<EncodedArch> archs = _archs;
  code syntax = _syntax;
  Pred pred = _pred;
}

/// Argument assembly format.
class ASMArgFormat {
  // Assembly printer code.
  // Example: `$_printer.printOperand($_self);`
  code asmPrinter = ?;
}

class PrintOperand : ASMArgFormat {
  let asmPrinter = "$_printer.printOperand($_self);";
}

//===----------------------------------------------------------------------===//
// Effects
//===----------------------------------------------------------------------===//

class Effect {
  // A string summary of the effect.
  string summary;
  // The code body for this effect. It has several special variables available:
  // - `$_op`, the current operation being processed.
  // - `$_effects`, the vector of effects collected for this instruction.
  code body;
}

//===----------------------------------------------------------------------===//
// Operands and modifiers
//===----------------------------------------------------------------------===//

// Class to represent a type that can be inferred.
class InferrableType<string _code> {
  // The code to infer the type of the operand. It has access to the following
  // variables:
  // - `$_builder`, the builder.
  // - `$_loc`, the location.
  // - $_self, the adaptor.
  // The code should produce the type as an expression.
  code inferType = _code;
}

// A type constraint that checks if the operand is of any of the specified types.
class AnyOperandOf<list<Type> types> : AnyTypeOf<types>, PrintOperand;

// A modifier attribute.
class ModAttr<Attr _attr> : PassthroughAttr<_attr>, ASMArgFormat {
}

// An unit modifier attribute.
class UnitModifier<string name, string _summary> : ModAttr<UnitAttr> {
  let summary = _summary;
  let asmPrinter = StrSubst<[{
    if ($_self)
      $_printer.printKeyword("$_name");
  }], [VarRepl<"_name", name>]>.result;
}

//===----------------------------------------------------------------------===//
// Instruction base class
//===----------------------------------------------------------------------===//

/// Helper class to get the optional type of a type.
class _GetOptionalType<TypeConstraint type, bit isOpt> {
  TypeConstraint optType = !if(isOpt, type, Optional<
    !if(!isa<Type>(type),
      !cast<Type>(type), Type<type.predicate, type.summary, type.cppType>)
  >);
}

/// Helper class to extract information from a list of dag arguments.
class InstructionTraits<dag arguments, dag results> {
  OpArgInfo argInfo = OpArgInfo<arguments>;
  OpArgInfo resInfo = OpArgInfo<results>;

  // Determine if the instruction requires attr-sized operand segments.
  bit requiresAttrSizedOperandSegments = !gt(argInfo.numOptOperands, 1);

  // Determine if the instruction requires attr-sized result segments.
  bit requiresAttrSizedResultSegments = !gt(resInfo.numOptOperands, 1);

  // Generate the traits for the instruction.
  list<Trait> traits = !listconcat([],
    !if(requiresAttrSizedOperandSegments, [AttrSizedOperandSegments], []),
    !if(requiresAttrSizedResultSegments, [AttrSizedResultSegments], [])
  );
}

/// Helper class to extract information from a list of dag arguments.
class InstOutsInfo<dag outputs>  : OpArgInfo<outputs> {
  // Create argument objects with optional types.
  list<TypeConstraint> typesAsOpt = !if(!empty(operands),
    /*true_value=*/[],
    /*false_value=*/!foreach(_op, operands, _GetOptionalType<_op.type, _op.isOpt>.optType)
  );
  // Generate result names by appending "_res" to argument names.
  list<string> resNames = !foreach(_op, operands, _op.name # "_res");
  dag instResults = !if(!gt(numOperands, 0),
    /*true_value=*/!dag(outs, typesAsOpt, resNames),
    /*false_value=*/(outs)
  );
}

/// Trait to provide instruction operand and result information.
class InstructionTrait : NativeOpTrait<"InstructionTrait"> {
  let cppNamespace = "::mlir::aster";
}

// A helper class to represent an instruction.
class Instruction<Dialect dialect, string mnemonic, list<Trait> opTraits = []>
    : Op<dialect, mnemonic, opTraits> {
  // A list of architectures this instruction is valid for, if empty, then
  // it's valid for all architectures.
  list<Arch> archs = [];
  // A list of encodings for this instruction.
  list<InstEnc> encodings = [];
  // A list of side effects of this instruction.
  list<Effect> effects = [];
  // A list of assembly syntaxes for this instruction.
  list<ASMString> asm_syntax = [];
  // A list of extra class declarations for this instruction.
  list<code> extraClassDecls = [];
  // The outputs of the instruction.
  dag outputs = (ins);
  // The inputs of the instruction.
  dag inputs = (ins);
  // The trailing arguments of the instruction.
  dag trailingArgs = (ins);
  // The trailing results of the instruction.
  dag trailingResults = (outs);
  // The arguments of the MLIR operation.
  let arguments = !con(outputs, inputs, trailingArgs);
  // The result outputs of the instruction.
  InstOutsInfo outputsInfo = InstOutsInfo<outputs>;
  // The results of the operation.
  let results = !con(outputsInfo.instResults, trailingResults);
  // The instruction operand and result information.
  InstructionTraits info = InstructionTraits<arguments, results>;
  // Generate the MLIR assembly format for the operation.
  bit genInstAssemblyFormat = 1;
  // The mnemonic of the instruction for assembly syntax.
  string asm_mnemonic = mnemonic;
  let traits = [
    DeclareOpInterfaceMethods<InstOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ] # opTraits # info.traits;
  let hasCustomAssemblyFormat = genInstAssemblyFormat;
  let extraClassDeclaration = !interleave(extraClassDecls, "\n");
}

#endif // ASTER_IR_INST_TD
