//===- RewriteUtils.h - Rewrite Pattern Utilities --------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for populating rewrite pattern sets with
// canonicalization patterns from dialects.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_REWRITEUTILS_H
#define ASTER_IR_REWRITEUTILS_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::aster {

/// Add canonicalization patterns for a single dialect (dialect-level and
/// operation-level) to the pattern set if the dialect is loaded.
template <typename DialectT>
void addCanonicalizationPatternsForDialect(MLIRContext *context,
                                           RewritePatternSet &patterns) {
  if (auto *dialect = context->getLoadedDialect<DialectT>()) {
    dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName opName :
         context->getRegisteredOperationsByDialect(dialect->getNamespace())) {
      opName.getCanonicalizationPatterns(patterns, context);
    }
  }
}

/// Add canonicalization patterns for a variadic pack of dialects to the
/// pattern set. Each dialect is loaded if available and contributes its
/// dialect-level and operation-level canonicalization patterns.
template <typename... DialectTs>
void addCanonicalizationPatterns(MLIRContext *context,
                                 RewritePatternSet &patterns) {
  (addCanonicalizationPatternsForDialect<DialectTs>(context, patterns), ...);
}

} // namespace mlir::aster

#endif // ASTER_IR_REWRITEUTILS_H
