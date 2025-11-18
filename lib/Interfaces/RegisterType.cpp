//===- RegisterType.cpp - RegisterType interface ----------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/RegisterType.h"

using namespace mlir;
using namespace mlir::aster;

llvm::raw_ostream &mlir::aster::operator<<(llvm::raw_ostream &os,
                                           const Register &reg) {
  if (reg.isRelocatable())
    return os << "?";
  return os << reg.getRegister();
}

llvm::raw_ostream &mlir::aster::operator<<(llvm::raw_ostream &os,
                                           const RegisterRange &range) {
  os << "[";
  if (range.begin().isRelocatable()) {
    os << "? + " << range.size();
  } else {
    os << range.begin() << " : " << range.end();
  }

  // Print alignment if it's different from the default alignment
  int expectedAlignment = defaultAlignment(range.size());
  if (range.alignment() != expectedAlignment) {
    os << " align " << range.alignment();
  }

  os << "]";
  return os;
}
FailureOr<Register> Register::parse(AsmParser &parser) {
  int regNumber;
  if (parser.parseInteger(regNumber))
    return failure();
  return Register(regNumber);
}

FailureOr<RegisterRange> RegisterRange::parse(AsmParser &parser) {
  // Parse opening bracket
  if (parser.parseLSquare())
    return failure();

  // Check for unallocated range (starts with '?')
  if (succeeded(parser.parseOptionalQuestion())) {
    // Parse '+'
    if (parser.parsePlus())
      return failure();

    // Parse num_registers
    int numRegisters;
    if (parser.parseInteger(numRegisters))
      return failure();

    // Parse optional alignment
    std::optional<int> alignment = std::nullopt;
    if (succeeded(parser.parseOptionalKeyword("align"))) {
      int alignValue;
      if (parser.parseInteger(alignValue))
        return failure();
      alignment = alignValue;
    }

    // Parse closing bracket
    if (parser.parseRSquare())
      return failure();

    return RegisterRange(Register(), numRegisters, alignment);
  }

  // Parse allocated range (begin : end)
  int begin;
  if (parser.parseInteger(begin))
    return failure();

  // Parse ':'
  if (parser.parseColon())
    return failure();

  // Parse end
  int end;
  if (parser.parseInteger(end))
    return failure();

  int numRegisters = end - begin; // right-exclusive

  // Parse optional alignment
  std::optional<int> alignment = std::nullopt;
  if (succeeded(parser.parseOptionalKeyword("align"))) {
    int alignValue;
    if (parser.parseInteger(alignValue))
      return failure();
    alignment = alignValue;
  }

  // Parse closing bracket
  if (parser.parseRSquare())
    return failure();

  return RegisterRange(Register(begin), numRegisters, alignment);
}

#include "aster/Interfaces/RegisterType.cpp.inc"
