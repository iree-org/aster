//===- Lexer.h - Lexer helpers ----------------------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_SUPPORT_LEXER_H
#define ASTER_SUPPORT_LEXER_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace mlir::aster {
namespace amdgcn {
/// Base class for lexers.
class Lexer {
public:
  Lexer(llvm::StringRef input, int line = 1, int col = 1)
      : currentPos(input), line(line), col(col) {}
  /// Returns true if the lexer has reached the end of the input.
  bool isAtEnd() const { return currentPos.empty(); }
  /// Get the current position in the input.
  llvm::StringRef getCurrentPos() const { return currentPos; }
  /// Get the current line number.
  int getLine() const { return line; }
  /// Get the current column number.
  int getCol() const { return col; }
  /// Get the current SMLoc.
  llvm::SMLoc getSMLoc() const {
    return llvm::SMLoc::getFromPointer(currentPos.data());
  }

  /// Lex an identifier [a-zA-Z_][a-zA-Z0-9_]*.
  FailureOr<llvm::StringRef> lexIdentifier();

  /// Lex a number.
  FailureOr<APInt> lexInt();

  /// Lex a float.
  FailureOr<APFloat> lexFloat();

  /// Consume the given string if it is present at the current position.
  LogicalResult consume(llvm::StringRef str);

  /// Get the current character.
  char currentChar() const {
    return currentPos.empty() ? '\0' : currentPos.front();
  }

  /// Consume whitespace characters.
  void consumeWhiteSpace() { skipWhitespace(currentPos, line, col); }

  /// Consume the current character.
  void consumeChar() {
    if (isAtEnd())
      return;
    if (currentChar() == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
    currentPos = currentPos.drop_front();
  }

protected:
  /// Skip whitespace characters.
  void skipWhitespace(llvm::StringRef &pos, int &line, int &col);

private:
  llvm::StringRef currentPos;
  int line;
  int col;
};
} // namespace amdgcn
} // namespace mlir::aster

#endif // ASTER_SUPPORT_LEXER_H
