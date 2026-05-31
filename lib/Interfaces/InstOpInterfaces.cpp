//===- InstOpInterfaces.cpp - Inst Op Interfaces ----------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/InstOpInterfaces.h"
#include "aster/Interfaces/RegisterType.h"

using namespace mlir;
using namespace mlir::aster;

#include "aster/Interfaces/InstOpInterfaces.cpp.inc"

LogicalResult
mlir::aster::detail::canonicalizeMovInstImpl(MovInstOpInterface mov,
                                             RewriterBase &rewriter) {
  RegisterTypeInterface dstTy =
      dyn_cast<RegisterTypeInterface>(mov.getDestOperand().getType());
  RegisterTypeInterface srcTy =
      dyn_cast<RegisterTypeInterface>(mov.getSrcOperand().getType());

  // Bail if either source or destination is not a register type.
  if (!srcTy || !dstTy)
    return failure();

  RegisterSemantics semantics = srcTy.getSemantics();
  // Bail if source and destination have different semantics.
  if (semantics != dstTy.getSemantics())
    return rewriter.notifyMatchFailure(
        mov, "source and destination have different register semantics");

  // We can remove the mov if the source and destination are the same register.
  // There are 2 cases:
  // 1. If allocated, it suffices to check if the types are the same.
  // 2. If unallocated or value, we need to check if the values are the same.

  if (semantics == RegisterSemantics::Allocated) {
    if (srcTy != dstTy) {
      return rewriter.notifyMatchFailure(
          mov,
          "source and destination have different allocated register types");
    }
    rewriter.eraseOp(mov);
    return success();
  }

  if (mov.getDestOperand().getValue() != mov.getSrcOperand().getValue()) {
    return rewriter.notifyMatchFailure(
        mov, "source and destination have different register values");
  }

  if (semantics == RegisterSemantics::Unallocated)
    rewriter.eraseOp(mov);
  else
    rewriter.replaceOp(mov, mov.getSrcOperand().getValue());

  return success();
}
