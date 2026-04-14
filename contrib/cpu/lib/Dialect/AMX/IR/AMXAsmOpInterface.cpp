// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/AMX/IR/Interfaces/AMXAsmOpInterface.h"

#include "aster/Dialect/AMX/IR/AMXDialect.h"
#include "aster/Dialect/X86/IR/X86Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "aster/Dialect/AMX/IR/Interfaces/AMXAsmOpInterface.cpp.inc"

using namespace mlir;
using namespace mlir::aster::amx;

::llvm::StringRef mlir::aster::amx::getPhysicalRegisterName(Value v) {
  Type t = v.getType();
  if (auto gpr = dyn_cast<::mlir::aster::x86::GprType>(t))
    return ::mlir::aster::x86::stringifyGprEnum(gpr.getReg());
  if (auto tile = dyn_cast<TileType>(t))
    return tile.getName().getValue();
  llvm_unreachable("expected !x86.gpr or !amx.tile value type");
}

//===----------------------------------------------------------------------===//
// Per-op printAsm implementations.
// Note: Uses AT&T operand order: rhs, lhs, dst.
//===----------------------------------------------------------------------===//

// TODO: Less manual C++ / more Tablegen stuff.

void LdTileCfgOp::printAsm(raw_ostream &os) {
  // The config block is emitted by the TranslateModule before the
  // function body as a per-function .rodata label `.Lcfg_<funcname>`.
  auto parentFunc = (*this)->getParentOfType<func::FuncOp>();
  os << (*this)->getName().stripDialect() << " .Lcfg_"
     << parentFunc.getSymName() << "(%rip)\n";
}

void TileLoaddOp::printAsm(raw_ostream &os) {
  os << (*this)->getName().stripDialect() << " (%"
     << getPhysicalRegisterName(getBase()) << ", %"
     << getPhysicalRegisterName(getStride()) << ", 1), %"
     << getPhysicalRegisterName(getResult()) << "\n";
}

void TdpBf16PsOp::printAsm(raw_ostream &os) {
  os << (*this)->getName().stripDialect() << " %"
     << getPhysicalRegisterName(getRhs()) << ", %"
     << getPhysicalRegisterName(getLhs()) << ", %"
     << getPhysicalRegisterName(getResult()) << "\n";
}

void TileStoredOp::printAsm(raw_ostream &os) {
  os << (*this)->getName().stripDialect() << " %"
     << getPhysicalRegisterName(getValue()) << ", (%"
     << getPhysicalRegisterName(getBase()) << ", %"
     << getPhysicalRegisterName(getStride()) << ", 1)\n";
}

void TileReleaseOp::printAsm(raw_ostream &os) {
  os << (*this)->getName().stripDialect() << "\n";
}
