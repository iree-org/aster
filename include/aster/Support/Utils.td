//===- Utils.td - Tblgen utils -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_SUPPORT_UTILS_TD
#define ASTER_SUPPORT_UTILS_TD

/// Substitute string `_value` for substring `_tgt`.
class SubstCase<string _tgt, string _value> {
  string tgt = _tgt;
  string value = _value;
}

/// Substitute integer `_value` for substring `_tgt`.
class IntSubstCase<string _tgt, int _value>
    : SubstCase<_tgt, !cast<string>(_value)>;

/// Substitute variable `$var` with `value`.
class VarRepl<string var, string value>
    : SubstCase<!strconcat("$", var), value>;

/// Substitute an int variable `$var` with `value`.
class IntRepl<string var, int value>
    : IntSubstCase<!strconcat("$", var), value>;

/// Substitute all occurrences of substrings in `cases` within `input`.
/// The cases are applied in order, i.e., earlier cases have higher priority.
/// Usage:
///   defvar result = StrSubst<"abcde", [SubstCase<"ab", "x">,
///                                      SubstCase<"cd", "y">]>.result;
///   // result == "xye"
///
class StrSubst<string input, list<SubstCase> cases> {
  string result =
      !foldl(input, cases, res, curr, !subst(curr.tgt, curr.value, res));
}

#endif // ASTER_SUPPORT_UTILS_TD
