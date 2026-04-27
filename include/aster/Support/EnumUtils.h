//===- EnumUtils.h - Typed enum utilities -----------------------*- C++ -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides TypedEnum, a type-safe enum wrapper.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_SUPPORT_ENUMUTILS_H
#define ASTER_SUPPORT_ENUMUTILS_H

#include "mlir/Support/TypeID.h"
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace mlir::aster {
namespace impl {
template <typename EnumTy>
inline constexpr int32_t getEnumNumBits() {
  if constexpr (std::is_enum_v<EnumTy>) {
    return std::numeric_limits<
        std::make_unsigned_t<std::underlying_type_t<EnumTy>>>::digits;
  }
  return 0;
}
} // namespace impl

/// Type-safe enum wrapper.
class TypedEnum {
public:
  template <typename EnumTy>
  static constexpr bool is_valid_enum_v = impl::getEnumNumBits<EnumTy>();

  TypedEnum() = default;
  TypedEnum(TypedEnum &&) = default;
  TypedEnum(const TypedEnum &) = default;
  TypedEnum &operator=(TypedEnum &&other) {
    value = std::move(other.value);
    typeID = std::move(other.typeID);
    return *this;
  }
  TypedEnum &operator=(const TypedEnum &other) {
    value = other.value;
    typeID = other.typeID;
    return *this;
  }
  template <typename EnumTy, std::enable_if_t<is_valid_enum_v<EnumTy>, int> = 0>
  TypedEnum &operator=(EnumTy value) {
    this->value = static_cast<int64_t>(value);
    this->typeID = mlir::TypeID::get<EnumTy>();
    return *this;
  }

  template <typename EnumTy, std::enable_if_t<is_valid_enum_v<EnumTy>, int> = 0>
  static bool classof(const TypedEnum &e) {
    return e.typeID == mlir::TypeID::get<EnumTy>();
  }

  /// Create a TypedEnum from a specific enum value.
  template <typename EnumTy, std::enable_if_t<is_valid_enum_v<EnumTy>, int> = 0>
  static TypedEnum get(EnumTy value) {
    return TypedEnum(static_cast<int64_t>(value), mlir::TypeID::get<EnumTy>());
  }

  bool operator==(const TypedEnum &other) const {
    return typeID == other.typeID && value == other.value;
  }
  bool operator!=(const TypedEnum &other) const { return !(*this == other); }

  /// Compares this enum against using three-way comparison semantics. Returns
  /// std::nullopt if the enums are of different types.
  std::optional<int> compare(const TypedEnum &other) const {
    if (typeID != other.typeID)
      return std::nullopt;
    if (value < other.value)
      return -1;
    if (value > other.value)
      return 1;
    return 0;
  }

  /// Returns the enum value as an integer.
  int64_t getValue() const { return value; }

  /// Returns the TypeID of the enum type.
  mlir::TypeID getTypeID() const { return typeID; }

  /// Try to get the enum value as a specific enum type. Returns std::nullopt if
  /// the type does not match.
  template <typename EnumTy, std::enable_if_t<is_valid_enum_v<EnumTy>, int> = 0>
  std::optional<EnumTy> getValueAs() const {
    if (typeID != mlir::TypeID::get<EnumTy>())
      return std::nullopt;
    return static_cast<EnumTy>(value);
  }

protected:
  TypedEnum(int64_t value, mlir::TypeID typeID)
      : value(value), typeID(typeID) {}
  int64_t value = 0;
  mlir::TypeID typeID;
};
} // namespace mlir::aster

#endif // ASTER_SUPPORT_ENUMUTILS_H
