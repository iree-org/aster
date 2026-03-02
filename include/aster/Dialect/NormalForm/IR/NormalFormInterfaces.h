//===- NormalFormInterfaces.h ---------------------------------------------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H
#define ASTER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#include "aster/Dialect/NormalForm/IR/NormalFormAttrInterfaces.h.inc"

namespace llvm {
template <>
struct PointerLikeTypeTraits<normalform::NormalFormAttrInterface>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline normalform::NormalFormAttrInterface
  getFromVoidPointer(void *p) {
    return normalform::NormalFormAttrInterface(
        mlir::Attribute::getFromOpaquePointer(p));
  }
};
} // namespace llvm

#endif // ASTER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H
