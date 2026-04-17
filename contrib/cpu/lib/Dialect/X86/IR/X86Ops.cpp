// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/AMX/IR/Interfaces/AMXAsmOpInterface.h"
#include "aster/Dialect/X86/IR/X86Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::aster::x86;

//===----------------------------------------------------------------------===//
// AT&T assembly printer for x86 instructions.
//
// Centralizes register formatting and operand conventions:
//   %reg           -- register operand
//   (%base)        -- simple memory operand
//   (%base, %idx, scale) -- indexed memory operand
//   ", "           -- operand separator
//
// Mirrors lib/Target/ASM/AsmPrinter in the AMDGCN backend.
//===----------------------------------------------------------------------===//

namespace {

/// Emit the bare register name (no %) for any x86 physical register type.
void emitRegName(raw_ostream &os, Value v) {
  llvm::TypeSwitch<Type, void>(v.getType())
      .Case<GprType>([&](auto gpr) { os << stringifyGprEnum(gpr.getReg()); })
      .Case<TmmType>([&](auto t) { os << "tmm" << t.getReg(); })
      .Case<XMMType>([&](auto t) { os << "xmm" << t.getReg(); })
      .Case<YMMType>([&](auto t) { os << "ymm" << t.getReg(); })
      .Case<ZMMType>([&](auto t) { os << "zmm" << t.getReg(); })
      .Default([](Type) { llvm_unreachable("unknown x86 register type"); });
}

/// AT&T assembly printer.
struct ATT {
  raw_ostream &os;

  ATT &inst(StringRef m) {
    os << m << " ";
    return *this;
  }
  ATT &reg(Value v) {
    os << "%";
    emitRegName(os, v);
    return *this;
  }
  ATT &mem(Value base) {
    os << "(%";
    emitRegName(os, base);
    os << ")";
    return *this;
  }
  ATT &mem(Value base, Value index, int64_t scale) {
    os << "(%";
    emitRegName(os, base);
    os << ", %";
    emitRegName(os, index);
    os << ", " << scale << ")";
    return *this;
  }
  ATT &sep() {
    os << ", ";
    return *this;
  }
  ATT &nl() {
    os << "\n";
    return *this;
  }
};

/// Extract the x86 instruction mnemonic from an MLIR op name.
/// "x86.avx.vfmadd231ps" -> "vfmadd231ps"
StringRef instrName(Operation *op) {
  StringRef name = op->getName().stripDialect();
  return name.substr(name.rfind('.') + 1);
}

} // namespace

//===----------------------------------------------------------------------===//
// AVX / AVX2 / AVX512 FMA ops
//===----------------------------------------------------------------------===//

namespace {
template <typename OpT>
void printFmaAsm(OpT op, raw_ostream &os) {
  ATT{os}
      .inst(instrName(op))
      .reg(op.getRhs())
      .sep()
      .reg(op.getLhs())
      .sep()
      .reg(op.getResult())
      .nl();
}
} // namespace

void AVXVFmaddOp::printAsm(raw_ostream &os) { printFmaAsm(*this, os); }
IsaVersion AVXVFmaddOp::getRequiredIsa() { return IsaVersion::avx; }

void AVX2VFmaddOp::printAsm(raw_ostream &os) { printFmaAsm(*this, os); }
IsaVersion AVX2VFmaddOp::getRequiredIsa() { return IsaVersion::avx2; }

void AVX512VFmaddOp::printAsm(raw_ostream &os) { printFmaAsm(*this, os); }
IsaVersion AVX512VFmaddOp::getRequiredIsa() { return IsaVersion::avx512; }

//===----------------------------------------------------------------------===//
// AVX / AVX2 / AVX512 load/store ops
//===----------------------------------------------------------------------===//

namespace {
template <typename OpT>
void printLoadAsm(OpT op, raw_ostream &os) {
  ATT{os}.inst(op.getInst()).mem(op.getBase()).sep().reg(op.getResult()).nl();
}
template <typename OpT>
void printStoreAsm(OpT op, raw_ostream &os) {
  ATT{os}.inst(op.getInst()).reg(op.getValue()).sep().mem(op.getBase()).nl();
}
} // namespace

void AVXLoadOp::printAsm(raw_ostream &os) { printLoadAsm(*this, os); }
IsaVersion AVXLoadOp::getRequiredIsa() { return IsaVersion::avx; }

void AVX2LoadOp::printAsm(raw_ostream &os) { printLoadAsm(*this, os); }
IsaVersion AVX2LoadOp::getRequiredIsa() { return IsaVersion::avx2; }

void AVX512LoadOp::printAsm(raw_ostream &os) { printLoadAsm(*this, os); }
IsaVersion AVX512LoadOp::getRequiredIsa() { return IsaVersion::avx512; }

void AVXStoreOp::printAsm(raw_ostream &os) { printStoreAsm(*this, os); }
IsaVersion AVXStoreOp::getRequiredIsa() { return IsaVersion::avx; }

void AVX2StoreOp::printAsm(raw_ostream &os) { printStoreAsm(*this, os); }
IsaVersion AVX2StoreOp::getRequiredIsa() { return IsaVersion::avx2; }

void AVX512StoreOp::printAsm(raw_ostream &os) { printStoreAsm(*this, os); }
IsaVersion AVX512StoreOp::getRequiredIsa() { return IsaVersion::avx512; }
