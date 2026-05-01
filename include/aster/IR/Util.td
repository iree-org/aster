//===- Util.td - Utility definitions -----------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_UTIL_TD
#define ASTER_IR_UTIL_TD

include "aster/Support/EnumBase.td"
include "aster/Support/Utils.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute constraints
//===----------------------------------------------------------------------===//

// A class to define an attribute based on input attribute.
class PassthroughAttr<Attr _base> : Attr<_base.predicate, _base.summary> {
  let storageType = _base.storageType;
  let returnType = _base.returnType;
  let convertFromStorage = _base.convertFromStorage;
  let constBuilderCall = _base.constBuilderCall;
  let defaultValue = _base.defaultValue;
  let valueType = _base.valueType;
  let isOptional = _base.isOptional;
  let baseAttr = _base.baseAttr;
  let cppNamespace = _base.cppNamespace;
  let description = _base.description;
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

class CPPType<string cppType, string summary> : Type<TruePred, summary, cppType> {
  let predicate = CPred<
    StrSubst<"::llvm::isa<$_type>($_self)", [VarRepl<"_type", cppType>]>.result
  >;
  let description = summary;
}

//===----------------------------------------------------------------------===//
// Operation reflection utilities
//===----------------------------------------------------------------------===//

class _OperandInfo<string _n, TypeConstraint _t, int index> {
  string name = _n;
  TypeConstraint type = _t;
  int argIndex = index;
}

// Helper class to represent an argument.
class OpOperandArg<string _n, TypeConstraint _t, int index> : _OperandInfo<_n, _t, index> {
  // Extract if the argument is optional.
  bit isOptional = !isa<Optional>(type);

  // Extract if the argument is variadic.
  bit isVariadic = !isa<Variadic>(type);

  // Extract if the argument is a variadic of a variadic.
  bit isVariadicOfVariadic = !isa<VariadicOfVariadic>(type);

  // Extract if the argument is optional, variadic, or a variadic of a variadic.
  bit isOpt = !or(isOptional, isVariadic, isVariadicOfVariadic);
}

// Helper class to extract information from a list of dag arguments.
class OpArgInfo<dag args> {
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

  // Helper to extract the operand arguments.
  list<_OperandInfo> _operands = !if(!empty(args),
    /*true_value=*/[],
    /*false_value=*/!filter(_op,
      !foreach(i, indices, _OperandInfo<names[i], !getdagarg<TypeConstraint>(args, i), i>),
      !initialized(_op.type)
    )
  );

  // The operand arguments.
  list<OpOperandArg> operands = !if(!empty(_operands),
    /*true_value=*/[],
    /*false_value=*/!foreach(_op, _operands, OpOperandArg<_op.name, _op.type, _op.argIndex>)
  );

  // The number of operand arguments.
  int numOperands = !size(operands);

  // Extract if the arguments are optional.
  list<int> isOpt = !if(!empty(operands),
    /*true_value=*/[],
    /*false_value=*/!foreach(_op, operands, _op.isOpt)
  );

  // Get the number of optional arguments.
  int numOptOperands = !foldl(0, isOpt, acc, v, !add(v, acc));
}

#endif // ASTER_IR_UTIL_TD
