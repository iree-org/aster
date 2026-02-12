//===- InstBase.td - Instruction base definitions -----------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
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

#ifndef ASTER_IR_INSTBASE_TD
#define ASTER_IR_INSTBASE_TD

include "aster/Support/Utils.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

/// Helper class to represent an argument.
class InstArg<string _n, TypeConstraint _t> {
  string name = _n;
  TypeConstraint type = _t;

  // Extract if the argument is optional.
  bit isOptional = !isa<Optional>(type);

  // Extract if the argument is variadic.
  bit isVariadic = !isa<Variadic>(type);

  // Extract if the argument is a variadic of a variadic.
  bit isVariadicOfVariadic = !isa<VariadicOfVariadic>(type);

  // Extract if the argument is optional, variadic, or a variadic of a variadic.
  bit isOpt = !or(isOptional, isVariadic, isVariadicOfVariadic);
}

/// Helper class to get the optional type of a type.
class _GetOptType<TypeConstraint type, bit isOpt> {
  TypeConstraint optType = !if(isOpt, type, Optional<
    !if(!isa<Type>(type),
      !cast<Type>(type), Type<type.predicate, type.summary, type.cppType>)
  >);
}

/// Helper class to extract information from a list of dag arguments.
class InstArgInfo<dag args> {
  // Generate indices for the args.
  list<int> indices = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!range(!size(args))
  );

  // Get the number of arguments.
  int numArgs = !size(indices);

  // Extract the names of the arguments.
  list<string> names = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, !getdagname(args, i))
  );

  // Extract the types of the arguments.
  list<TypeConstraint> types = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, !getdagarg<TypeConstraint>(args, i))
  );

  // Create argument objects.
  list<InstArg> argList = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, InstArg<names[i], types[i]>)
  );

  // Create argument objects with optional types.
  list<TypeConstraint> typesAsOpt = !if(!empty(argList),
    /*true_value=*/[],
    /*false_value=*/!foreach(arg, argList, _GetOptType<arg.type, arg.isOpt>.optType)
  );

  // Extract if the arguments are optional.
  list<int> isOpt = !if(!empty(argList),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, indices, argList[i].isOpt)
  );

  // Get the number of optional arguments.
  int numOptArgs = !foldl(0, isOpt, acc, v, !add(v, acc));
}

/// Helper class to extract information from a list of dag arguments.
class InstInfo<dag inputs, dag outputs, dag leadingArgs, int leadingResultsSize> {
  InstArgInfo inputsInfo = InstArgInfo<inputs>;
  InstArgInfo outputsInfo = InstArgInfo<outputs>;

  // Get the number of optional arguments.
  int numOptArgs = !add(inputsInfo.numOptArgs, outputsInfo.numOptArgs);

  // Determine if the instruction requires attr-sized operand segments.
  bit requiresAttrSizedOperandSegments = !gt(numOptArgs, 1);

  // Determine if the instruction requires attr-sized result segments.
  bit requiresAttrSizedResultSegments = !gt(outputsInfo.numArgs, 1);

  // Generate the traits for the instruction.
  list<Trait> traits = !listconcat([],
    !if(requiresAttrSizedOperandSegments, [AttrSizedOperandSegments], []),
    !if(requiresAttrSizedResultSegments, [AttrSizedResultSegments], [])
  );

  // Generate result names by appending "_res" to argument names.
  list<string> resNames = !foreach(n, outputsInfo.names, n#"_res");
  dag instResults = !if(!gt(outputsInfo.numArgs, 0),
    /*true_value=*/!dag(outs, outputsInfo.typesAsOpt, resNames),
    /*false_value=*/(outs)
  );

  // Generate indices for the args.
  list<int> leadingIndices = !if(!empty(leadingArgs),
    /*true_value=*/[],
    /*false_value=*/!range(!size(leadingArgs))
  );

  // Extract the types of the arguments.
  list<bit> isLeadingOperand = !if(!empty(leadingArgs),
    /*true_value=*/[],
    /*false_value=*/!foreach(i, leadingIndices, !initialized(!getdagarg<TypeConstraint>(leadingArgs, i)))
  );

  // The number of leading operands.
  int numLeadingOperands = !if(!empty(leadingArgs),
    /*true_value=*/0,
    /*false_value=*/!foldl(0, isLeadingOperand, acc, v, !add(!if(v, 1, 0), acc))
  );

  // The number of instruction results.
  int numInstResults = !size(instResults);

  // The number of leading results.
  int numLeadingResults = leadingResultsSize;

}

/// Trait to provide instruction operand and result information.
class InstTrait<InstInfo info> : NativeOpTrait<"InstTrait"> {
  let cppNamespace = "::mlir::aster";
  let dependentTraits = [ConditionallySpeculatable, MemoryEffectsOpInterface];
  let extraConcreteClassDeclaration = StrSubst<[{
    /// The number of input operands.
    static constexpr int64_t kInsSize = $kInsSize;

    /// The number of output operands.
    static constexpr int64_t kOutsSize = $kOutsSize;

    /// The number of leading operands.
    static constexpr int64_t kLeadingOperandsSize = $kLeadingOperandsSize;

    /// The number of leading results.
    static constexpr int64_t kLeadingResultsSize = $kLeadingResultsSize;

    /// Get the instruction output operands.
    OperandRange getInstOuts() {
      return getInstOutsImpl(*this);
    }

    /// Get the instruction input operands.
    OperandRange getInstIns() {
      return getInstInsImpl(*this);
    }

    /// Get the instruction results.
    ResultRange getInstResults() {
      return getInstResultsImpl(*this);
    }
  }], [
    IntRepl<"kInsSize", info.inputsInfo.numArgs>,
    IntRepl<"kOutsSize", info.outputsInfo.numArgs>,
    IntRepl<"kLeadingOperandsSize", info.numLeadingOperands>,
    IntRepl<"kLeadingResultsSize", info.numLeadingResults>
  ]>.result;
}

/// Trait to provide pure value semantics and memory effects for instructions.
def PureInstTrait : NativeOpTrait<"InstTrait"> {
  let cppNamespace = "::mlir::aster";
  let extraConcreteClassDeclaration = [{
    /// Returns the speculatability of the instruction.
    Speculation::Speculatability getSpeculatability() {
      return mlir::aster::detail::getInstSpeculatabilityImpl(*this);
    }
    /// Returns the memory effects of the instruction.
    void getEffects(
        SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
            &effects) {
      return mlir::aster::detail::getInstEffectsImpl(*this, effects);
    }
  }];
}

/// Trait to provide pure value semantics and memory effects for instructions.
def PureInst : TraitList<[
  ConditionallySpeculatable,
  MemoryEffectsOpInterface,
  PureInstTrait
]>;

/// Trait to provide infer type methods for instructions.
def InstInferTypeTrait : NativeOpTrait<"InstTrait"> {
  let cppNamespace = "::mlir::aster";
  let extraConcreteClassDeclaration = [{
  //===------------------------------------------------------------------===//
  // InferType Methods
  //===------------------------------------------------------------------===//
  static ::llvm::LogicalResult inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
  let extraConcreteClassDefinition = [{
  ::llvm::LogicalResult $cppClass::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
    return ::mlir::aster::detail::inferTypesImpl<$cppClass>(operands, attributes, properties, regions, inferredReturnTypes);
  }
  }];
}

/// Trait to provide infer type methods for instructions.
def InstInferType : TraitList<[
  InferTypeOpInterface,
  InstInferTypeTrait
]>;

// Base class for all instructions. Users must not use the Op fields set by this
// class directly, instead they should let the InstOp class handle them. For example,
// don't set the arguments and results fields directly, instead set the outputs
// and inputs fields.
class InstOperation<Dialect dialect, string mnemonic, list<Trait> opTraits = []>
  : Op<dialect, mnemonic, opTraits> {
  // The output operands of the instruction.
  dag outputs = (ins);

  // The input operands of the instruction.
  dag inputs = (ins);

  // Leading arguments that are not part of the input or output operands.
  dag leadingArguments = (ins);

  // Trailing arguments that are not part of the input or output operands.
  dag trailingArguments = (ins);

  // Leading results that are not part of the instruction results.
  dag leadingResults = (outs);

  // Trailing results that are not part of the instruction results.
  dag trailingResults = (outs);

  // The instruction operand and result information.
  InstInfo info = InstInfo<inputs, outputs, leadingArguments, !size(leadingResults)>;

  // Helper with the arguments of the instruction.
  dag instArguments = !con(leadingArguments, outputs, inputs, trailingArguments);

  // The instruction results.
  dag deducedInstResults = info.instResults;

  // Helper with the results of the instruction.
  dag instResults = !con(leadingResults, deducedInstResults, trailingResults);

  // The arguments of the operation.
  let arguments = instArguments;

  // The results of the operation.
  let results = instResults;

  // Whether to generate the instruction verifier.
  bit genInstVerifier = 1;
  let hasVerifier = genInstVerifier;

  // The traits of the operation.
  let traits = !listconcat(info.traits, opTraits, [InstTrait<info>]);
}

#endif // ASTER_IR_INSTBASE_TD
