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

#include "aster/Interfaces/RegisterSemantics.cpp.inc"

using namespace mlir;
using namespace mlir::aster;

llvm::raw_ostream &mlir::aster::operator<<(llvm::raw_ostream &os,
                                           const Register &reg) {
  switch (reg.getSemantics()) {
  case RegisterSemantics::Allocated:
    return os << reg.getRegister();
  case RegisterSemantics::Unallocated:
    return os << "?";
  case RegisterSemantics::Value:
    return os << "*";
  }
  return os;
}

llvm::raw_ostream &mlir::aster::operator<<(llvm::raw_ostream &os,
                                           const RegisterRange &range) {
  if (range.size() == 1 && range.alignment() == 1)
    return os << range.begin();
  os << "[";

  switch (range.getSemantics()) {
  case RegisterSemantics::Allocated:
    os << range.begin() << " : " << range.end();
    break;
  case RegisterSemantics::Unallocated:
    os << "? : ? + " << range.size();
    break;
  case RegisterSemantics::Value:
    os << "? + " << range.size();
    break;
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
  // Check for unallocated register '?'
  if (succeeded(parser.parseOptionalQuestion()))
    return Register::getAsUnallocated();

  // Check for value register '*'
  if (succeeded(parser.parseOptionalStar()))
    return Register::getAsValue();

  // Otherwise parse allocated register number
  int regNumber;
  if (parser.parseInteger(regNumber))
    return failure();
  return Register(regNumber);
}

FailureOr<RegisterRange> RegisterRange::parse(AsmParser &parser) {
  // Try to parse `[`, if failed, parse a single register.
  if (failed(parser.parseOptionalLSquare())) {
    FailureOr<Register> reg = Register::parse(parser);
    if (failed(reg))
      return failure();
    return RegisterRange(reg.value(), 1);
  }

  // Check for unallocated or value range (starts with '?')
  if (succeeded(parser.parseOptionalQuestion())) {
    // Check if it's unallocated range (? : ? + num_regs)
    if (succeeded(parser.parseOptionalColon())) {
      // Parse '?'
      if (parser.parseQuestion())
        return failure();

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

      return RegisterRange(Register::getAsUnallocated(), numRegisters,
                           alignment);
    }

    // Otherwise it's a value range (? + num_regs)
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

    return RegisterRange(Register::getAsValue(), numRegisters, alignment);
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
