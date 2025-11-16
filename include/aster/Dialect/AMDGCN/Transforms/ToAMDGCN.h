//===- ToAMDGCN.h - Convert to AMDGCN -------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_TRANSFORM_TOAMDGCN_H
#define ASTER_DIALECT_AMDGCN_TRANSFORM_TOAMDGCN_H

#include "aster/Transforms/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class FrozenRewritePatternSet;
namespace aster {
namespace amdgcn {
/// The type converter used for instruction selection.
struct ToAMDGCNConverter : TypeConverter, Builder, FuncTypeConverter {
  ToAMDGCNConverter(MLIRContext &context);

  /// Get the MLIR context.
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context;
};

/// Base class for instruction selection patterns.
struct ToAMDGCNPatternBase {
  ToAMDGCNPatternBase(ToAMDGCNConverter &converter) : converter(converter) {}
  /// Create an allocation of the given register type.
  Value createAllocation(RewriterBase &rewriter, Location loc,
                         Type regTy) const;

  /// Get or split the given value range into multiple registers.
  ValueRange getOrSplitRange(RewriterBase &rewriter, Location loc,
                             ValueRange values) const;

protected:
  ToAMDGCNConverter &converter;
};

/// Base class for instruction selection patterns.
template <typename ConcreteType>
class OpToAMDGCNPattern : public OpConversionPattern<ConcreteType>,
                          protected ToAMDGCNPatternBase {
public:
  using OpConversionPattern<ConcreteType>::OpConversionPattern;
  using Base = OpToAMDGCNPattern;
  using Op = ConcreteType;
  using OpAdaptor = typename ConcreteType::Adaptor;
  OpToAMDGCNPattern(ToAMDGCNConverter &converter, PatternBenefit benefit = 1)
      : OpConversionPattern<ConcreteType>(converter, converter.getContext(),
                                          benefit),
        ToAMDGCNPatternBase(converter) {}
};

/// Populate the given pattern list with instruction selection patterns.
void populateToAMDGCNPatterns(ToAMDGCNConverter &converter,
                              RewritePatternSet &patterns,
                              ConversionTarget &target);
} // namespace amdgcn
} // namespace aster
} // namespace mlir

#endif // ASTER_DIALECT_AMDGCN_TRANSFORM_TOAMDGCN_H
