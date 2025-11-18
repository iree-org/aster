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

  /// Convert the function signature.
  static FunctionType convertFunctionSignature(const TypeConverter &converter,
                                               FunctionOpInterface funcOp,
                                               SignatureConversion &result) {
    return convertFunctionSignatureImpl(
        converter, cast<FunctionType>(funcOp.getFunctionType()), result);
  }

private:
  /// Convert the function signature.
  static FunctionType
  convertFunctionSignatureImpl(const TypeConverter &converter,
                               FunctionType funcTy,
                               SignatureConversion &result);
};

/// Populate function conversion patterns.
void populateFuncConversionPatterns(TypeConverter &converter,
                                    ConversionTarget &target,
                                    RewritePatternSet &patterns);
} // namespace aster
} // namespace mlir

#endif // ASTER_TRANSFORMS_UTILS_H
