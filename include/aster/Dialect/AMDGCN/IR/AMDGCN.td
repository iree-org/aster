//===-- AMDGCN.td - AMDGCN target definitions -----------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines AMDGCN target definitions.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_AMDGCN_TD
#define ASTER_AMDGCN_TD

include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNEnumAttrs.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNModifiers.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.td"
include "aster/Dialect/AMDGCN/IR/AMDGCNInstOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/Constraints.td"

//===----------------------------------------------------------------------===//
// Inst Op Helpers
//===----------------------------------------------------------------------===//

/// Utility class to indicate that no builder should be generated.
class SkipBuilderTag : OpBuilder<(ins)>;
def SkipBuilder : SkipBuilderTag;

/// Utility class to indicate that the op builder from the op definition
// should be used.
class UseOpBuilderTag : OpBuilder<(ins)>;
def UseOpBuilder : UseOpBuilderTag;

/// Utility class to configure various properties of an operation.
class OpProps<list<OpBuilder> _builders = [], bit _skipDefaultBuilders = 0,
              string _assemblyFormat = "", bit _hasVerifier = 0,
              bit _hasCanonicalizer = 0, bit _hasCanonicalizeMethod = 0,
              bit _hasFolder = 0, code _extraClassDeclaration = "",
              code _extraClassDefinition = ""> {
  list<OpBuilder> builders = _builders;
  bit skipDefaultBuilders = _skipDefaultBuilders;
  string assemblyFormat = _assemblyFormat;
  bit hasVerifier = _hasVerifier;
  bit hasCanonicalizer = _hasCanonicalizer;
  bit hasCanonicalizeMethod = _hasCanonicalizeMethod;
  bit hasFolder = _hasFolder;
  code extraClassDeclaration = _extraClassDeclaration;
  code extraClassDefinition = _extraClassDefinition;
}

/// Optional type match.
class _OptTypeMatch<string lhs, string rhs>
    : OptionalTypesMatchWith<lhs#" and "#rhs#"must be equal", lhs, rhs,
                             "$_self">;

/// Helper class to represent an argument with its name and optionality.
class _Arg<string _n, bit opt> {
  string name = _n;
  bit isOptional = opt;
}

/// Helper class to determined whether the optional arguments are at the end
// of a list.
class _AreOptsAtEnd<list<_Arg> args, list<_Arg> nonOpt, list<_Arg> opt> {
  list<_Arg> _t0 = nonOpt # opt;
  list<bit> _t1 = !foreach(i, !range(args), !eq(args[i], _t0[i]));
  bit result = !foldl(1, _t1, acc, v, !and(v, acc));
}

/// Helper class to extract information from a list of dag arguments.
class _ArgInfo<dag args, bit isIns = 0> {
  // Generate indices for the args.
  list<int> indices = !range(!size(args));
  // Get the number of arguments.
  int numArgs = !size(indices);
  // Extract names, types, and optionality.
  list<string> names = !foreach(i, indices, !getdagname(args, i));
  list<TypeConstraint> types =
      !foreach(i, indices, !getdagarg<TypeConstraint>(args, i));
  list<bit> isOpt = !foreach(
      i, indices, !or(!isa<Optional>(types[i]), !isa<Variadic>(types[i])));
  // Create argument objects.
  list<_Arg> argList = !foreach(i, indices, _Arg<names[i], isOpt[i]>);
  // Get fixed and optional arguments.
  list<_Arg> fixedArgs = !filter(a, argList, !not(a.isOptional));
  list<_Arg> optArgs = !filter(a, argList, a.isOptional);
  // Get argument formats.
  code fixedArgFmt = !interleave(!foreach(a, fixedArgs, "$"#a.name), " `,` ");
  code optArgFmt = !interleave(
    !foreach(a, optArgs, "(`"#a.name#"` `=` $"#a.name # "^)?"), " ");
  // Generate full argument format.
  code argFmt = !if(!gt(numArgs, 0),
    !interleave([!if(isIns, "`ins`", "`outs`"), fixedArgFmt, optArgFmt], " "), "");
  // Generate argument count code.
  code argCount = StrSubst<[{static constexpr size_t $key = $size;}], [
      VarRepl<"key", !if(isIns, "kInsSize", "kOutsSize")>,
      IntRepl<"size", !size(indices)>
    ]>.result;
  // Check if optional arguments are at the end.
  bit areOptsAtEnd = _AreOptsAtEnd<argList, fixedArgs, optArgs>.result;
}

/// Trait to provide instruction operand information.
class _InstOpTrait<dag ins, dag outs> : NativeOpTrait<"InstOpTrait"> {
  let cppNamespace = "::mlir::aster::amdgcn";
  _ArgInfo insInfo = _ArgInfo<ins, 1>;
  _ArgInfo outsInfo = _ArgInfo<outs, 0>;
  code inferTypeDeclImpl = !if(!gt(outsInfo.numArgs, 0), [{
    //===------------------------------------------------------------------===//
    // InferType Methods
    //===------------------------------------------------------------------===//
    static ::llvm::LogicalResult inferReturnTypes(
      ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
      ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }], [{}]);
  code inferTypeDefImpl = !if(!gt(outsInfo.numArgs, 0), [{
    //===------------------------------------------------------------------===//
    // InferType Methods
    //===------------------------------------------------------------------===//
    ::llvm::LogicalResult $cppClass::inferReturnTypes(
      ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> loc,
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr attrs,
      ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
      ::llvm::SmallVectorImpl<::mlir::Type> &types) {
      return inferTypesImpl<$cppClass>(context, loc,
        $cppClass::Adaptor(operands, attrs, properties, regions), types);
    }
  }], [{}]);
  let extraConcreteClassDeclaration = StrSubst<[{
    $ins_size
    $outs_size
    $infer_type
    //===------------------------------------------------------------------===//
    // InstOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Get the instruction output operands.
    MutableArrayRef<OpOperand> getInstOutsMutable() {
      return getOperation()->getOpOperands().take_front(
        getNumOuts(*this, kOutsSize));
    }
    /// Get the instruction input operands.
    MutableArrayRef<OpOperand> getInstInsMutable() {
      return getOperation()->getOpOperands().slice(
        getNumOuts(*this, kOutsSize), getNumIns(*this, kOutsSize, kInsSize));
    }
  }], [
    VarRepl<"ins_size", insInfo.argCount>,
    VarRepl<"outs_size", outsInfo.argCount>,
    VarRepl<"infer_type", inferTypeDeclImpl>
  ]>.result;
  let extraConcreteClassDefinition = inferTypeDefImpl;
}

/// Helper class to generate assembly format string for an operation.
class _OpInfo<dag inputs, dag outputs> {
  // Argument info for inputs and outputs.
  _ArgInfo inputsInfo = _ArgInfo<inputs, 1>;
  _ArgInfo outputsInfo = _ArgInfo<outputs, 0>;
  int numArgs = !add(!size(inputsInfo.argList), !size(outputsInfo.argList));
  int numFixedArgs =
      !add(!size(inputsInfo.fixedArgs), !size(outputsInfo.fixedArgs));
  int numOptArgs = !add(!size(inputsInfo.optArgs), !size(outputsInfo.optArgs));
  // Generate outputs and inputs format strings.
  code typesFmt = !if(!gt(numArgs, 0), "`:` type(operands)", "");
  string fmt = StrSubst<[{
   custom<Opcode>($opcode) $outputs $inputs attr-dict $types
  }], [
    VarRepl<"outputs", outputsInfo.argFmt>,
    VarRepl<"inputs", inputsInfo.argFmt>,
    VarRepl<"types", typesFmt>
  ]>.result;
  // Generate result names by appending "_res" to argument names.
  list<string> resNames = !foreach(n, outputsInfo.names, n#"_res");
  dag results = !dag(outs, outputsInfo.types, resNames);
  // Generate traits for type matching.
  list<Trait> traits =
      !listconcat([AMDGCNInstOpInterface, _InstOpTrait<inputs, outputs>],
                  !foreach(
                      i, outputsInfo.indices,
                      !if(outputsInfo.isOpt[i],
                          _OptTypeMatch<outputsInfo.names[i], resNames[i]>,
                          AllTypesMatch<[outputsInfo.names[i], resNames[i]]>)),
                  !if(!gt(outputsInfo.numArgs, 0), [InferTypeOpInterface], []),
                  !if(!gt(numOptArgs, 1), [AttrSizedOperandSegments], []));
}

/// Helper class to define an AMDGCN instruction operations.
class AutoInstOp<string mnemonic, list<Trait> _traits = [],
                 bit _genInstVerifier = 1, OpProps opProps = OpProps<>>
    : InstOp<mnemonic, []> {
  let cppNamespace = "::mlir::aster::amdgcn::inst";
  // The inputs of the instruction.
  dag inputs = (ins);
  // The outputs of the instruction.
  dag outputs = (ins);
  // The attributes of the instruction.
  dag attrs = (ins);
  // Create the arguments dag by concatenating all the dags.
  let arguments = !con((ins AMDGCN_InstAttr:$opcode), outputs, inputs, attrs);
  _OpInfo opInfo = _OpInfo<inputs, outputs>;
  // Create the results dag from the outputs.
  let results = opInfo.results;
  // The assembly format string.
  let assemblyFormat =
      !if(!empty(opProps.assemblyFormat), opInfo.fmt, opProps.assemblyFormat);
  let hasCustomAssemblyFormat = !not(!initialized(assemblyFormat));
  let traits = opInfo.traits#_traits;
  // Whether to generate the instruction verifier using amdgcn-tblgen.
  let genInstVerifier = _genInstVerifier;
  let hasVerifier = !if(genInstVerifier, 1, opProps.hasVerifier);

  // Use OpProps to configure various properties of the op.
  let builders = opProps.builders;
  let skipDefaultBuilders = opProps.skipDefaultBuilders;
  let hasCanonicalizer = opProps.hasCanonicalizer;
  let hasCanonicalizeMethod = opProps.hasCanonicalizeMethod;
  let hasFolder = opProps.hasFolder;
  let extraClassDeclaration = opProps.extraClassDeclaration;
  let extraClassDefinition = opProps.extraClassDefinition;
  // Check pre-conditions.
  assert opInfo.inputsInfo.areOptsAtEnd,
    "Optional input arguments must be at the end of the argument list: " #!repr(inputs);
  assert opInfo.outputsInfo.areOptsAtEnd,
    "Optional output arguments must be at the end of the argument list: " #!repr(outputs);
}

/// Helper class to get the appropriate op builder.
class _GetBuilder<OpBuilder instBuilder, OpBuilder _opBuilder> {
  assert !not(!isa<UseOpBuilderTag>(_opBuilder)),
    "Op builder cannot be UseOpBuilder";
  OpBuilder result = !cond(
    !isa<SkipBuilderTag>(instBuilder) : ?,
    !isa<UseOpBuilderTag>(instBuilder) : !cond(
      !isa<SkipBuilderTag>(_opBuilder) : ?,
      true : _opBuilder),
    true : instBuilder);
}

/// An instruction case.
class InstCase<string _mnemonic, list<AsmVariant> _asmFormat,
               dag _constraints = (ins),
               list<Target> _targets = ?, string _description = "",
               OpBuilder _cppBuilder = UseOpBuilder,
               OpBuilder _pythonBuilder = UseOpBuilder,
               string _identifier = ""> {
  string mnemonic = _mnemonic;
  string description = _description;
  dag constraints = _constraints;
  OpBuilder cppBuilder = _cppBuilder;
  OpBuilder pythonBuilder = _pythonBuilder;
  list<Target> targets = _targets;
  string identifier =
    !if(!empty(_identifier), snakeCaseToCamelCase<_mnemonic>.ret, _identifier);
  list<AsmVariant> asmFormat = _asmFormat;
}

//===----------------------------------------------------------------------===//
// Inst helpers
//===----------------------------------------------------------------------===//

/// Defines a default C++ builder for an instruction op.
class DefaultCppBuilder<dag params> : OpBuilder<params, !if(
!gt(!size(params), 0), [{
  return $_create($_builder, $_loc, $_opcode, $_args);
}], [{
  return $_create($_builder, $_loc, $_opcode);
}])>;

/// Defines a default python builder for an instruction op.
class DefaultPyBuilder<dag params> : OpBuilder<params, !if(
!gt(!size(params), 0), [{
  return $_create(opcode=$_opcode, $_args, $_lastArgs)
}], [{
  return $_create(opcode=$_opcode, $_lastArgs)
}])>;

/// Shorthand for C++ values in the instruction builders.
defvar CppValue = "::mlir::Value";

/// Shorthand for Python values in the instruction builders.
defvar PyValue = "$_ir.Value";

/// Helper for C++/python integer arguments on builders.
class IntArg<int v> : CArg<"int", !cast<string>(v)>;

class IntOrValueArg<int v> : CArg<"$_typing.Union[int, $_ir.Value]", !cast<string>(v)>;

/// Helper for python boolean arguments on builders.
class PyBool<bit v> : CArg<"bool", !if(v, "True", "False")>;

/// A helper for when an instruction has a single assembly variant.
class SingleAsmVariant<string format> : AsmVariant<TruePred, format>;

//===----------------------------------------------------------------------===//
// Inst operands helpers
//===----------------------------------------------------------------------===//

/// Type with description.
/// This is hack to use use the description field of Type to provide custom error
/// messages for instruction operand constraints.
class TypeWithDescription<Pred condition, string summary, string _desc = "",
                        string cppType = "::mlir::Type">
  : Type<condition, summary, cppType> {
  let description = _desc;
}

/// Register operand constraint. It check the number of registers and alignment.
class RegisterOperand<
    list<Type> validRegTypes, int numRegs, int alignment = -1, bit optional = 0
  > : InstConstraint<> {
  code cPred = !subst("$alignment", !cast<string>(alignment),
                      !subst("$numRegs", !cast<string>(numRegs),
                      "isValidRegisterType($_self, $numRegs, $alignment)"));
  code desc = !subst("$alignment", !cast<string>(alignment),
                      !subst("$numRegs", !cast<string>(numRegs),
                      "$_argName operand must have $numRegs registers for $_mnemonic, got "));
  let constraints = [
    !if(optional, Optional<AnyTypeOf<validRegTypes>>, AnyTypeOf<validRegTypes>),
    !if(optional,
      Optional<TypeWithDescription<CPred<cPred>, desc, "getRangeSize($_self)", "AMDGCNRegisterTypeInterface">>,
      TypeWithDescription<CPred<cPred>, desc, "getRangeSize($_self)", "AMDGCNRegisterTypeInterface">)
  ];
}

/// Register or Imm operand constraint. It check the number of registers and alignment.
class RegImmOperand<Type validType, int numRegs, bit optional = 0> :
    InstConstraint<> {
  code cPred = !subst("$numRegs", !cast<string>(numRegs),
                      "isValidRegImmType($_self, $numRegs)");
  code desc = !subst("$numRegs", !cast<string>(numRegs),
                     "$_argName operand must have $numRegs or be an Imm registers for $_mnemonic, got ");
  let constraints = [
    !if(optional, Optional<validType>, validType),
    !if(optional,
      Optional<TypeWithDescription<CPred<cPred>, desc, "getRangeSize($_self)", "Type">>,
      TypeWithDescription<CPred<cPred>, desc, "getRangeSize($_self)", "Type">)
  ];
}

/// SGPR operand constraint.
class SGPROperand<int numRegs, bit optional = 0, int alignment = -1> :
    RegisterOperand<[SGPRType, SGPRRangeType], numRegs, alignment, optional> {
}

/// VGPR operand constraint.
class VGPROperand<int numRegs, bit optional = 0, int alignment = -1> :
    RegisterOperand<[VGPRType, VGPRRangeType], numRegs, alignment, optional> {
}

/// ACC operand constraint.
class AccOperand<int numRegs, bit optional = 0, int alignment = -1> :
    RegisterOperand<[VGPRType, AGPRType, VGPRRangeType, AGPRRangeType], numRegs, alignment, optional> {
}

/// Addr operand constraint.
class AddrOperand<int numRegs, bit optional = 0, int alignment = -1> :
    RegisterOperand<[SGPRType, VGPRType, SGPRRangeType, VGPRRangeType], numRegs, alignment, optional> {
}

/// The optional value must not be present.
def NotPresent : InstConstraint<[
  TypeConstraint<
    CPred<"$_self == nullptr">, "expected `$_argName` to not be present">
]>;

/// The optional value must be present.
def Present : InstConstraint<[
  TypeConstraint<
    CPred<"$_self != nullptr">, "expected `$_argName` to be present">
]>;

/// A register or immediate operand that can be either SGPR or VGPR type
def VIntSrc : RegOrImm<[SGPRType, VGPRType], [I32]>;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

def InvalidCase : EnumCaseInfo<"invalid", "Invalid">;

//===----------------------------------------------------------------------===//
// ISA Versions
//===----------------------------------------------------------------------===//

def CDNA3 : ISAVersion<"cdna3"> { let description = "CDNA 3 ISA version"; }

def RDNA4 : ISAVersion<"rdna4"> { let description = "RDNA 4 ISA version"; }

def IsaVersions
    : AutoI32EnumCases<"ISAVersion",
                       "AMDGCN ISA Versions", [InvalidCase, CDNA3, RDNA4]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def IsaVersionAttr : AMDGCN_EnumAttr<"IsaVersion", "isa", IsaVersions>;

//===----------------------------------------------------------------------===//
// Target definitions
//===----------------------------------------------------------------------===//

def GFX940 : Target<"gfx940", CDNA3> { let description = "GFX940 target"; }

def GFX942 : Target<"gfx942", CDNA3> { let description = "GFX942 target"; }

def GFX1201 : Target<"gfx1201", RDNA4> { let description = "GFX1201 target"; }

def AMDGCNTargets
    : AutoI32EnumCases<
          "Target", "AMDGCN Targets", [InvalidCase, GFX940, GFX942, GFX1201]> {
  let cppNamespace = "::mlir::aster::amdgcn";
}

def AMDGCNTargetAttr : AMDGCN_EnumAttr<"Target", "target", AMDGCNTargets>;

#endif // ASTER_AMDGCN_TD
