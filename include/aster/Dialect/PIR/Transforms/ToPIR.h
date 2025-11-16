//===- ToPIR.h - Convert to PIR -------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TRANSFORM_TOPIR_H
#define ASTER_TRANSFORM_TOPIR_H

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
namespace pir {
/// State information for instruction selection.
class ConvertToPIRState : public Builder {
public:
  ConvertToPIRState() = delete;
  ConvertToPIRState(ConvertToPIRState &&) = default;
  ~ConvertToPIRState();
  /// Create an instruction selection state for the given root operation.
  static FailureOr<ConvertToPIRState> create(Operation *op);

  /// Check if the given value is known to be thread-uniform.
  bool isThreadUniform(Value value) const;

  /// Get the size of the given type according to the data layout.
  int64_t getTypeSize(Type type) const;
  int64_t getTypeSizeInBits(Type type) const;

private:
  using Builder::Builder;
  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// The type converter used for instruction selection.
struct ToPIRConverter : TypeConverter {
  ToPIRConverter(ConvertToPIRState &state);
  /// Get the size of the given type according to the data layout.
  int64_t getTypeSize(Type type) const { return state->getTypeSize(type); }
  int64_t getTypeSizeInBits(Type type) const {
    return state->getTypeSizeInBits(type);
  }

  /// Check if the given value is known to be thread-uniform.
  bool isThreadUniform(Value value) const {
    return state->isThreadUniform(value);
  }

  /// Get the MLIR context.
  MLIRContext *getContext() const { return state->getContext(); }

  /// Return the instruction selection state.
  const ConvertToPIRState &getState() const {
    assert(state && "State is not initialized");
    return *state;
  }
  /// Get the index type used in this conversion.
  Type getIndexType() const { return indexType; }

private:
  ConvertToPIRState *state;
  Type indexType;
};

/// Base class for instruction selection patterns.
struct ToPIRPatternBase {
  ToPIRPatternBase(ToPIRConverter &converter) : converter(converter) {}
  /// Create an alloca of the given register type.
  Value createAlloca(RewriterBase &rewriter, Location loc, Type regTy) const;

protected:
  ToPIRConverter &converter;
};

/// Base class for instruction selection patterns.
template <typename ConcreteType>
class OpToPIRPattern : public OpConversionPattern<ConcreteType>,
                       protected ToPIRPatternBase {
public:
  using OpConversionPattern<ConcreteType>::OpConversionPattern;
  using Base = OpToPIRPattern;
  using Op = ConcreteType;
  using OpAdaptor = typename ConcreteType::Adaptor;
  OpToPIRPattern(ToPIRConverter &converter, PatternBenefit benefit = 1)
      : OpConversionPattern<ConcreteType>(converter, converter.getContext(),
                                          benefit),
        ToPIRPatternBase(converter) {}
};

/// Populate the given pattern list with instruction selection patterns.
void populateToPIRPatterns(ToPIRConverter &converter,
                           RewritePatternSet &patterns,
                           ConversionTarget &target);
} // namespace pir
} // namespace aster
} // namespace mlir

#endif // ASTER_TRANSFORM_TOPIR_H
