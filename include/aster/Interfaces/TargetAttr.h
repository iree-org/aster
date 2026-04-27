//===- TargetAttr.h - Target attribute interface ----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetAttr attribute interface and the Target and
// TargetFamily typed enum types.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_TARGETATTR_H
#define ASTER_INTERFACES_TARGETATTR_H

#include "aster/Support/EnumUtils.h"
#include "mlir/IR/Attributes.h"

namespace mlir::aster {
/// Identifies a specific compilation target.
struct TargetArch : TypedEnum {
  using TypedEnum::TypedEnum;
  using TypedEnum::operator=;
};

/// Identifies a family of compilation targets.
struct TargetFamily : TypedEnum {
  using TypedEnum::TypedEnum;
  using TypedEnum::operator=;
};

/// Identifies a target encoding for an instruction.
struct Encoding : TypedEnum {
  using TypedEnum::TypedEnum;
};

/// Identifies a specific target and encoding for an instruction.
struct EncodedArch {
  EncodedArch(TargetFamily targetFamily, Encoding encoding)
      : targetFamily(targetFamily), encoding(encoding) {}
  TargetFamily targetFamily;
  Encoding encoding;
};
} // namespace mlir::aster

#include "aster/Interfaces/TargetAttr.h.inc"

#endif // ASTER_INTERFACES_TARGETATTR_H
