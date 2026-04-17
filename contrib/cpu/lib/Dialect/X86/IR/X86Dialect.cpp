// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/X86/IR/X86Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::aster::x86;

#include "aster/Dialect/X86/IR/X86Dialect.cpp.inc"
#include "aster/Dialect/X86/IR/X86Enums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aster/Dialect/X86/IR/X86Types.cpp.inc"

void X86Dialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aster/Dialect/X86/IR/X86Types.cpp.inc"
      >();
}
