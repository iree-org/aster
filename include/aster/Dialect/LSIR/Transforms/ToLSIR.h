//===- ToLSIR.h - Convert to LSIR -----------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TRANSFORM_TOLSIR_H
#define ASTER_TRANSFORM_TOLSIR_H

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
namespace lsir {
/// State information for instruction selection.
/// TODO: Use ABIanalysis here.
class ConvertToLSIRState : public Builder {
public:
  ConvertToLSIRState() = delete;
  ConvertToLSIRState(ConvertToLSIRState &&) = default;
  ~ConvertToLSIRState();
  /// Create an instruction selection state for the given root operation.
  static FailureOr<ConvertToLSIRState> create(Operation *op);

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
struct ToLSIRConverter : TypeConverter {
  ToLSIRConverter(ConvertToLSIRState &state);
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
  const ConvertToLSIRState &getState() const {
    assert(state && "State is not initialized");
    return *state;
  }
  /// Get the index type used in this conversion.
  Type getIndexType() const { return indexType; }

private:
  ConvertToLSIRState *state;
  Type indexType;
};

/// Base class for instruction selection patterns.
struct ToLSIRPatternBase {
  ToLSIRPatternBase(ToLSIRConverter &converter) : converter(converter) {}
  /// Create an alloca of the given register type.
  Value createAlloca(RewriterBase &rewriter, Location loc, Type regTy) const;

protected:
  ToLSIRConverter &converter;
};

/// Base class for instruction selection patterns.
template <typename ConcreteType>
class OpToLSIRPattern : public OpConversionPattern<ConcreteType>,
                        protected ToLSIRPatternBase {
public:
  using OpConversionPattern<ConcreteType>::OpConversionPattern;
  using Base = OpToLSIRPattern;
  using Op = ConcreteType;
  using OpAdaptor = typename ConcreteType::Adaptor;
  OpToLSIRPattern(ToLSIRConverter &converter, PatternBenefit benefit = 1)
      : OpConversionPattern<ConcreteType>(converter, converter.getContext(),
                                          benefit),
        ToLSIRPatternBase(converter) {}
};

/// Populate the given pattern list with instruction selection patterns.
void populateToLSIRPatterns(ToLSIRConverter &converter,
                            RewritePatternSet &patterns,
                            ConversionTarget &target);
} // namespace lsir
} // namespace aster
} // namespace mlir

#endif // ASTER_TRANSFORM_TOLSIR_H
