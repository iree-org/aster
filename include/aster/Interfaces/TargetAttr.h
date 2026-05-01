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
  /// Create a Encoding from a specific enum value.
  template <typename EnumTy,
            std::enable_if_t<TargetArch::is_valid_enum_v<EnumTy>, int> = 0>
  static TargetArch get(EnumTy value) {
    return TargetArch(static_cast<int64_t>(value), mlir::TypeID::get<EnumTy>());
  }
};

/// Identifies a family of compilation targets.
struct TargetFamily : TypedEnum {
  using TypedEnum::TypedEnum;
  using TypedEnum::operator=;
  /// Create a Encoding from a specific enum value.
  template <typename EnumTy,
            std::enable_if_t<TargetFamily::is_valid_enum_v<EnumTy>, int> = 0>
  static TargetFamily get(EnumTy value) {
    return TargetFamily(static_cast<int64_t>(value),
                        mlir::TypeID::get<EnumTy>());
  }
};

/// Identifies a target encoding for an instruction.
struct Encoding : TypedEnum {
  using TypedEnum::TypedEnum;
  using TypedEnum::operator=;
  /// Create a Encoding from a specific enum value.
  template <typename EnumTy,
            std::enable_if_t<Encoding::is_valid_enum_v<EnumTy>, int> = 0>
  static Encoding get(EnumTy value) {
    return Encoding(static_cast<int64_t>(value), mlir::TypeID::get<EnumTy>());
  }
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
