//===- CodeGen.h - Code Generation ----------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_CODEGEN_CODEGEN_H
#define ASTER_CODEGEN_CODEGEN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>
#include <cstdint>

namespace mlir {
class FrozenRewritePatternSet;
namespace aster {
/// State information for instruction selection.
/// TODO: Use ABIanalysis here.
class ConvertCodeGenState : public Builder {
public:
  ConvertCodeGenState() = delete;
  ConvertCodeGenState(ConvertCodeGenState &&) = default;
  ~ConvertCodeGenState();
  /// Create an instruction selection state for the given root operation.
  static FailureOr<ConvertCodeGenState> create(Operation *op);

  /// Check if the given value is known to be thread-uniform.
  bool isThreadUniform(Value value) const;

  /// Get the size of the given type according to the data layout.
  int64_t getTypeSize(Type type) const;
  int64_t getTypeSizeInBits(Type type) const;

  /// Get the register constraint attribute for the given value.
  Attribute getRegisterConstraint(Value value) const;

private:
  using Builder::Builder;
  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// The type converter used for instruction selection.
struct CodeGenConverter : TypeConverter {
  CodeGenConverter(ConvertCodeGenState &state);
  /// Get the size of the given type according to the data layout.
  int64_t getTypeSize(Type type) const { return state->getTypeSize(type); }
  int64_t getTypeSizeInBits(Type type) const {
    return state->getTypeSizeInBits(type);
  }

  /// Check if the given value is known to be thread-uniform.
  bool isThreadUniform(Value value) const {
    auto cOp =
        dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
    while (cOp) {
      if (cOp.getInputs().size() != 1)
        break;
      value = cOp.getInputs().front();
      cOp = dyn_cast_if_present<UnrealizedConversionCastOp>(
          value.getDefiningOp());
    }
    return state->isThreadUniform(value);
  }

  /// Get the MLIR context.
  MLIRContext *getContext() const { return state->getContext(); }

  /// Return the instruction selection state.
  const ConvertCodeGenState &getState() const {
    assert(state && "State is not initialized");
    return *state;
  }
  /// Get the index type used in this conversion.
  Type getIndexType() const { return indexType; }

private:
  ConvertCodeGenState *state;
  Type indexType;
};

/// Base class for instruction selection patterns.
struct CodeGenPatternBase {
  CodeGenPatternBase(CodeGenConverter &converter) : converter(converter) {}
  /// Create an alloca of the given register type.
  Value createAlloca(RewriterBase &rewriter, Location loc, Type regTy) const;

protected:
  CodeGenConverter &converter;
};

/// Base class for instruction selection patterns.
template <typename ConcreteType>
class OpCodeGenPattern : public OpConversionPattern<ConcreteType>,
                         protected CodeGenPatternBase {
public:
  using OpConversionPattern<ConcreteType>::OpConversionPattern;
  using Base = OpCodeGenPattern;
  using Op = ConcreteType;
  using OpAdaptor = typename ConcreteType::Adaptor;
  OpCodeGenPattern(CodeGenConverter &converter, PatternBenefit benefit = 1)
      : OpConversionPattern<ConcreteType>(converter, converter.getContext(),
                                          benefit),
        CodeGenPatternBase(converter) {}
};
} // namespace aster
} // namespace mlir

#endif // ASTER_CODEGEN_CODEGEN_H
