//===- Utils.h ------------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TRANSFORMS_UTILS_H
#define ASTER_TRANSFORMS_UTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace aster {
/// Utility struct for function type conversion.
struct FuncTypeConverter {
  using SignatureConversion = TypeConverter::SignatureConversion;
  /// Convert a function type.
  static FunctionType convertFuncType(const TypeConverter &converter,
                                      FunctionType type) {
    SignatureConversion result(type.getNumInputs());
    return convertFunctionSignatureImpl(converter, type, result);
  }
  static FunctionType convertFuncType(const TypeConverter &converter,
                                      FunctionType type,
                                      SignatureConversion &result) {
    return convertFunctionSignatureImpl(converter, type, result);
  }

  /// Convert the function signature.
  static FunctionType convertFunctionSignature(const TypeConverter &converter,
                                               FunctionOpInterface funcOp,
                                               SignatureConversion &result) {
    return convertFuncType(
        converter, cast<FunctionType>(funcOp.getFunctionType()), result);
  }

private:
  /// Convert the function signature.
  static FunctionType
  convertFunctionSignatureImpl(const TypeConverter &converter,
                               FunctionType funcTy,
                               SignatureConversion &result);
};

/// Generic operation conversion pattern that converts operations by converting
/// their result types and recreating them.
template <typename Op>
class GenericOpConversion : public OpConversionPattern<Op> {
public:
  using OpAdaptor = typename OpConversionPattern<Op>::OpAdaptor;
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (this->getTypeConverter()->isLegal(op.getOperation()))
      return failure();
    SmallVector<Type> rTys;
    if (failed(
            this->getTypeConverter()->convertTypes(op->getResultTypes(), rTys)))
      return failure();
    auto newOp = Op::create(rewriter, op.getLoc(), rTys, adaptor.getOperands(),
                            op.getProperties(),
                            op->getDiscardableAttrDictionary().getValue());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/// Get the ABI attribute from the given operation.
FunctionType getABI(Operation *op, FunctionType defaultType);

/// Set the ABI attribute on the given operation.
void setABI(Operation *op, FunctionType type);

/// Populate arithmetic conversion patterns.
void populateArithConversionPatterns(TypeConverter &converter,
                                     ConversionTarget &target,
                                     RewritePatternSet &patterns);

/// Populate function conversion patterns.
void populateFuncConversionPatterns(TypeConverter &converter,
                                    ConversionTarget &target,
                                    RewritePatternSet &patterns);

/// Populate ptr dialect conversion patterns.
void populatePtrConversionPatterns(TypeConverter &converter,
                                   ConversionTarget &target,
                                   RewritePatternSet &patterns);

/// Populate SCF conversion patterns.
void populateScfConversionPatterns(TypeConverter &converter,
                                   ConversionTarget &target,
                                   RewritePatternSet &patterns);
} // namespace aster
} // namespace mlir

#endif // ASTER_TRANSFORMS_UTILS_H
