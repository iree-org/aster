//===- RegisterTypes.h - Aster register-types ------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the aster register-like types.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_INTERFACES_REGISTERTYPE_H
#define ASTER_INTERFACES_REGISTERTYPE_H

#include "aster/Interfaces/ResourceInterfaces.h"
#include "aster/Support/API.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/Hashing.h"

#include <cstdint>
#include <optional>
#include <tuple>

#include "aster/Interfaces/RegisterSemantics.h.inc"

namespace mlir::aster {
/// Represents a single register.
class ASTER_EXPORTED Register {
public:
  Register() = default;
  explicit Register(int16_t number)
      : regNumber(number), semantics(RegisterSemantics::Allocated) {}

  /// Check if the register is relocatable.
  bool isRelocatable() const {
    return semantics != RegisterSemantics::Allocated;
  }

  /// Check if the register is valid.
  bool isValid() const { return isRelocatable() || getRegister() >= 0; }

  /// Get the register semantics.
  RegisterSemantics getSemantics() const { return semantics; }

  /// Get the register number.
  int16_t getRegister() const {
    assert(semantics == RegisterSemantics::Allocated &&
           "register is not allocated");
    return regNumber;
  }

  /// Get a register with the given offset.
  Register getWithOffset(int16_t offset) const {
    if (isRelocatable())
      return Register(semantics);
    return Register(regNumber + offset);
  }

  /// Get an unallocated register.
  static Register getAsUnallocated() {
    return Register(RegisterSemantics::Unallocated);
  }

  /// Get a value register.
  static Register getAsValue() { return Register(RegisterSemantics::Value); }

  bool operator==(const Register &other) const {
    return regNumber == other.regNumber && semantics == other.semantics;
  }

  /// Compare registers by number.
  bool operator<(const Register &other) const {
    return std::tie(semantics, regNumber) <
           std::tie(other.semantics, other.regNumber);
  }

  /// Hash a register.
  friend llvm::hash_code hash_value(const Register &reg) {
    return llvm::hash_combine(reg.regNumber, reg.semantics);
  }

  /// Parse a register from the given asm parser.
  static FailureOr<aster::Register> parse(AsmParser &parser);

private:
  explicit Register(RegisterSemantics sem) : semantics(sem) {}
  int16_t regNumber = -1;
  RegisterSemantics semantics = RegisterSemantics::Value;
};

/// Get the default alignment for a given size.
/// Returns size if it's a power of 2, otherwise the next power of 2.
inline int16_t defaultAlignment(int16_t size) {
  if (size <= 0)
    return 1;
  if ((size & (size - 1)) == 0)
    return size;
  // Find the next power of 2
  int16_t result = 1;
  while (result < size)
    result <<= 1;
  return result;
}

/// Represents a range of registers.
class ASTER_EXPORTED RegisterRange {
public:
  RegisterRange() = default;
  RegisterRange(Register begin, int16_t size,
                std::optional<int16_t> alignment = std::nullopt)
      : regBegin(begin), rangeSize(size),
        indexAlignment(alignment.value_or(defaultAlignment(size))) {}

  /// Get the beginning register of the range.
  Register begin() const { return regBegin; }

  /// Get the ending register of the range (right-exclusive).
  Register end() const {
    return regBegin.isRelocatable()
               ? regBegin
               : Register(regBegin.getRegister() + rangeSize);
  }

  /// Returns the size of the range.
  int16_t size() const { return rangeSize; }

  /// Returns the alignment of the range.
  int16_t alignment() const { return indexAlignment; }

  /// Get the register semantics.
  RegisterSemantics getSemantics() const { return regBegin.getSemantics(); }

  /// Get the register range as an unallocated range.
  RegisterRange getAsUnallocatedRange() const {
    return RegisterRange(Register::getAsUnallocated(), size(), indexAlignment);
  }

  /// Get the register range as a value range.
  RegisterRange getAsValueRange() const {
    return RegisterRange(regBegin.getAsValue(), size(), indexAlignment);
  }

  /// Check if the register range is equal to another register range.
  bool operator==(const RegisterRange &other) const {
    return regBegin == other.regBegin && rangeSize == other.rangeSize &&
           indexAlignment == other.indexAlignment;
  }

  /// Compare register ranges.
  bool operator<(const RegisterRange &other) const {
    return std::make_tuple(regBegin, rangeSize, indexAlignment) <
           std::make_tuple(other.regBegin, other.rangeSize,
                           other.indexAlignment);
  }

  /// Hash a register range.
  friend llvm::hash_code hash_value(const RegisterRange &range) {
    return llvm::hash_combine(range.regBegin, range.rangeSize,
                              range.indexAlignment);
  }

  /// Parse a register range from the given asm parser.
  static FailureOr<aster::RegisterRange> parse(AsmParser &parser);

private:
  /// The beginning register of the range.
  Register regBegin;

  /// The size of the range.
  int16_t rangeSize = 1;

  /// The alignment of the range.
  int16_t indexAlignment = 1;
};

/// Print a Register to a raw_ostream.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Register &reg);

/// Print a RegisterRange to a raw_ostream.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const RegisterRange &range);
} // namespace mlir::aster

namespace mlir {
template <>
struct FieldParser<aster::Register> {
  static FailureOr<aster::Register> parse(AsmParser &parser) {
    return aster::Register::parse(parser);
  }
};
template <>
struct FieldParser<aster::RegisterRange> {
  static FailureOr<aster::RegisterRange> parse(AsmParser &parser) {
    return aster::RegisterRange::parse(parser);
  }
};
} // namespace mlir

#include "aster/Interfaces/RegisterType.h.inc"

#endif // ASTER_INTERFACES_REGISTERTYPE_H
