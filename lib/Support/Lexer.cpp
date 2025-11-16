//===- Lexer.cpp - Lexer helpers --------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Support/Lexer.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

/// Skip whitespace characters.
void Lexer::skipWhitespace(llvm::StringRef &pos, int &line, int &col) {
  while (!pos.empty() && std::isspace(pos.front())) {
    if (pos.front() == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
    pos = pos.drop_front();
  }
}

/// Consume the given string if it is present at the current position.
LogicalResult Lexer::consume(llvm::StringRef str) {
  if (str.empty())
    return success();
  skipWhitespace(currentPos, line, col);
  if (!currentPos.starts_with(str))
    return failure();
  currentPos = currentPos.drop_front(str.size());
  col += str.size();
  return success();
}

/// Lex an identifier [a-zA-Z_][a-zA-Z0-9_]*.
FailureOr<llvm::StringRef> Lexer::lexIdentifier() {
  skipWhitespace(currentPos, line, col);

  if (currentPos.empty())
    return failure();

  // Check if first character is valid (letter or underscore)
  if (!std::isalpha(currentPos.front()) && currentPos.front() != '_')
    return failure();
  size_t len = 1;
  while (len < currentPos.size() &&
         (std::isalnum(currentPos[len]) || currentPos[len] == '_')) {
    ++len;
  }
  llvm::StringRef identifier = currentPos.take_front(len);
  currentPos = currentPos.drop_front(len);
  col += len;
  return identifier;
}

/// Lex a number.
FailureOr<APInt> Lexer::lexInt() {
  skipWhitespace(currentPos, line, col);

  if (currentPos.empty())
    return failure();

  bool isNegative = false;
  llvm::StringRef remaining = currentPos;

  // Handle sign
  if (remaining.front() == '-') {
    isNegative = true;
    remaining = remaining.drop_front();
  } else if (remaining.front() == '+') {
    remaining = remaining.drop_front();
  }

  if (remaining.empty() || !std::isdigit(remaining.front()))
    return failure();

  // Determine base
  unsigned radix = 10;
  if (remaining.starts_with("0x") || remaining.starts_with("0X")) {
    radix = 16;
    remaining = remaining.drop_front(2);
  } else if (remaining.starts_with("0b") || remaining.starts_with("0B")) {
    radix = 2;
    remaining = remaining.drop_front(2);
  } else if (remaining.starts_with("0o") || remaining.starts_with("0O")) {
    radix = 8;
    remaining = remaining.drop_front(2);
  }

  // Find the length of the number
  size_t len = 0;
  while (len < remaining.size()) {
    char c = remaining[len];
    if (radix == 16 && std::isxdigit(c)) {
      ++len;
    } else if (radix == 10 && std::isdigit(c)) {
      ++len;
    } else if (radix == 8 && c >= '0' && c <= '7') {
      ++len;
    } else if (radix == 2 && (c == '0' || c == '1')) {
      ++len;
    } else {
      break;
    }
  }
  if (len == 0)
    return failure();
  llvm::StringRef numStr = remaining.take_front(len);
  // Parse the number
  APInt result;
  if (numStr.getAsInteger(radix, result))
    return failure();

  if (isNegative)
    result = -result;

  // Update position
  size_t totalLen =
      (currentPos.data() + currentPos.size()) - remaining.data() + len;
  currentPos = currentPos.drop_front(totalLen);
  col += totalLen;
  return result;
}

/// Lex a float.
FailureOr<APFloat> Lexer::lexFloat() {
  skipWhitespace(currentPos, line, col);
  if (currentPos.empty())
    return failure();
  llvm::StringRef remaining = currentPos;
  size_t len = 0;
  bool hasDecimalPoint = false;
  bool hasExponent = false;
  bool hasDigits = false;
  // Handle sign
  if (remaining[len] == '-' || remaining[len] == '+') {
    ++len;
  }
  // Parse mantissa
  while (len < remaining.size()) {
    char c = remaining[len];
    if (std::isdigit(c)) {
      hasDigits = true;
      ++len;
    } else if (c == '.' && !hasDecimalPoint && !hasExponent) {
      hasDecimalPoint = true;
      ++len;
    } else if ((c == 'e' || c == 'E') && !hasExponent && hasDigits) {
      hasExponent = true;
      ++len;

      // Handle exponent sign
      if (len < remaining.size() &&
          (remaining[len] == '+' || remaining[len] == '-')) {
        ++len;
      }
    } else if (c == 'f' || c == 'F') {
      // Float suffix
      ++len;
      break;
    } else {
      break;
    }
  }
  if (!hasDigits || (!hasDecimalPoint && !hasExponent))
    return failure();

  llvm::StringRef floatStr = remaining.take_front(len);

  // Remove trailing 'f' or 'F' if present
  if (floatStr.ends_with("f") || floatStr.ends_with("F"))
    floatStr = floatStr.drop_back();

  // Parse the float
  APFloat result(APFloat::IEEEdouble());
  llvm::Expected<APFloat::opStatus> status =
      result.convertFromString(floatStr, APFloat::rmNearestTiesToEven);
  if (!status) {
    llvm::consumeError(status.takeError());
    return failure();
  }

  if (*status != APFloat::opOK && *status != APFloat::opInexact)
    return failure();

  // Update position
  currentPos = currentPos.drop_front(len);
  col += len;

  return result;
}
