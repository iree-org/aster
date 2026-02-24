//===- PrefixedOstream.h - raw_ostream with line prefix ---------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_SUPPORT_PREFIXEDOSTREAM_H
#define ASTER_SUPPORT_PREFIXEDOSTREAM_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace mlir::aster {

/// raw_ostream subclass that prints a prefix at the start of each line, with
/// optional indentation control.
class raw_prefixed_ostream : public llvm::raw_ostream {
public:
  raw_prefixed_ostream(llvm::raw_ostream &os, llvm::StringRef prefix = "")
      : os(os), prefix(prefix) {
    SetUnbuffered();
  }

  /// Set the prefix string. Safe to call between writes.
  void setPrefix(llvm::StringRef p) { prefix = p; }

  /// Returns the underlying raw_ostream.
  llvm::raw_ostream &getOStream() const { return os; }

  /// Decrease indentation by indentSize. Returns *this for chaining.
  raw_prefixed_ostream &unindent(int spaces = 2) {
    currentIndent = std::max(currentIndent - std::max(0, spaces), 0);
    return *this;
  }

  /// Set indentation level (number of spaces). Returns *this for chaining.
  raw_prefixed_ostream &indent(int spaces = 2) {
    currentIndent += std::max(0, spaces);
    return *this;
  }

  /// Return current indentation level.
  int getIndent() const { return currentIndent; }

private:
  void write_impl(const char *ptr, size_t size) final;

  uint64_t current_pos() const final { return os.tell(); }

  void printLinePrefix();

  llvm::raw_ostream &os;
  llvm::StringRef prefix;
  int currentIndent = 0;
  bool atStartOfLine = true;
};

inline void raw_prefixed_ostream::printLinePrefix() {
  if (!prefix.empty())
    os << prefix;
  for (int i = 0; i < currentIndent; ++i)
    os << ' ';
}

inline void raw_prefixed_ostream::write_impl(const char *ptr, size_t size) {
  llvm::StringRef str(ptr, size);

  while (!str.empty()) {
    size_t idx = str.find('\n');
    if (idx == llvm::StringRef::npos) {
      if (atStartOfLine) {
        printLinePrefix();
        atStartOfLine = false;
      }
      os << str;
      break;
    }

    llvm::StringRef line = str.substr(0, idx);
    if (atStartOfLine) {
      printLinePrefix();
      atStartOfLine = false;
    }
    os << line << '\n';
    atStartOfLine = true;
    str = str.substr(idx + 1);
  }
}
} // namespace mlir::aster

#endif // ASTER_SUPPORT_PREFIXEDOSTREAM_H
