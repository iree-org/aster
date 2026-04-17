// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aster/Dialect/AMX/Target/TranslateToAsm.h"

#include "aster/Dialect/AMX/IR/AMXDialect.h"
#include "aster/Dialect/AMX/IR/Interfaces/AMXAsmOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::aster::amx;

namespace {

/// Minimal x86_64 AT&T asm emitter for contrib/cpu AMX modules.
struct TranslateModuleImpl {
  TranslateModuleImpl(ModuleOp module, raw_ostream &os)
      : module(module), os(os) {}

  /// Translate the module to assembly.
  LogicalResult translate();

private:
  /// Emit the given function (tilecfg + prologue + body + epilogue).
  LogicalResult emitFunc(func::FuncOp func, size_t funcIndex);
  /// Emit a .rodata block containing the 64-byte AMX TILECFG derived
  /// from the `!amx.tile<tmmN, ...>` types referenced by `func`.
  LogicalResult emitFuncTileCfg(func::FuncOp func);
  /// Emit the SysV prologue: .text / .globl / .p2align / .type + label.
  LogicalResult emitFuncPrologue(func::FuncOp func);
  /// Emit the SysV epilogue: .Lfunc_end<N> label + .size directive.
  LogicalResult emitFuncEpilogue(func::FuncOp func, size_t funcIndex);
  /// Emit a single basic block.
  LogicalResult emitBlock(Block &block);
  /// Emit a single operation by dispatching through `AMXAsmOpInterface`.
  LogicalResult emitOperation(Operation &op);

  /// The module being translated.
  ModuleOp module;
  /// The output stream, with indentation under our control.
  raw_indented_ostream os;
};

/// Derive the AMX TILECFG bytes for a function from the tile types used in its
/// body. Fails if two ops reference the same tmm index with incompatible
/// shapes.
///
/// See:
///   llvm/lib/Target/X86/X86PreTileConfig.cpp
///   llvm/lib/Target/X86/X86TileConfig.cpp L148-L165.
static FailureOr<SmallVector<uint8_t, 64>>
buildTileCfgBytes(func::FuncOp func) {
  struct TileInfo {
    int64_t rows = -1;
    int64_t colsb = -1;
  };
  SmallVector<TileInfo, 8> tiles(8);

  auto visit = [&](TileType t) -> LogicalResult {
    int64_t idx = static_cast<int64_t>(t.getReg());
    StringRef name = ::mlir::aster::x86::stringifyTmmEnum(t.getReg());
    int64_t rowBytes =
        t.getCols() * (t.getElementType().getIntOrFloatBitWidth() / 8);
    if (rowBytes > 64)
      return func.emitError() << "amx tile " << name << " column-byte size "
                              << rowBytes << " exceeds 64";
    if (t.getRows() > 16)
      return func.emitError() << "amx tile " << name << " row count "
                              << t.getRows() << " exceeds 16";
    TileInfo &slot = tiles[idx];
    if (slot.rows != -1 && (slot.rows != t.getRows() || slot.colsb != rowBytes))
      return func.emitError()
             << "amx tile " << name
             << " declared with conflicting shapes in this function";
    slot.rows = t.getRows();
    slot.colsb = rowBytes;
    return success();
  };

  WalkResult walk = func.walk([&](Operation *op) -> WalkResult {
    for (Type t : op->getOperandTypes())
      if (auto tile = dyn_cast<TileType>(t))
        if (failed(visit(tile)))
          return WalkResult::interrupt();
    for (Type t : op->getResultTypes())
      if (auto tile = dyn_cast<TileType>(t))
        if (failed(visit(tile)))
          return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (walk.wasInterrupted())
    return failure();

  SmallVector<uint8_t, 64> bytes(64, 0);
  bytes[0] = 1; // palette_id = 1
  for (int i = 0; i < 8; ++i) {
    if (tiles[i].rows == -1)
      continue;
    bytes[16 + 2 * i] = static_cast<uint8_t>(tiles[i].colsb & 0xff);
    bytes[16 + 2 * i + 1] = static_cast<uint8_t>((tiles[i].colsb >> 8) & 0xff);
    bytes[48 + i] = static_cast<uint8_t>(tiles[i].rows);
  }
  return bytes;
}

LogicalResult TranslateModuleImpl::emitFuncTileCfg(func::FuncOp func) {
  FailureOr<SmallVector<uint8_t, 64>> bytes = buildTileCfgBytes(func);
  if (failed(bytes))
    return failure();
  StringRef name = func.getSymName();
  os.indent();
  os << ".section .rodata\n";
  os << ".p2align 6\n";
  os.unindent();
  os << ".Lcfg_" << name << ":\n";
  os.indent();
  for (int i = 0; i < 64; ++i)
    os << ".byte " << static_cast<int>((*bytes)[i]) << "\n";
  os.unindent();
  return success();
}

LogicalResult TranslateModuleImpl::emitFuncPrologue(func::FuncOp func) {
  StringRef name = func.getSymName();
  os.indent();
  os << ".text\n";
  os << ".globl " << name << "\n";
  os << ".p2align 4\n";
  os << ".type " << name << ",@function\n";
  os.unindent();
  os << name << ":\n";
  return success();
}

LogicalResult TranslateModuleImpl::emitFuncEpilogue(func::FuncOp func,
                                                    size_t funcIndex) {
  StringRef name = func.getSymName();
  os << ".Lfunc_end" << funcIndex << ":\n";
  os.indent();
  os << ".size " << name << ", .Lfunc_end" << funcIndex << "-" << name << "\n";
  os.unindent();
  return success();
}

LogicalResult TranslateModuleImpl::emitBlock(Block &block) {
  os.indent();
  for (Operation &op : block)
    if (failed(emitOperation(op)))
      return failure();
  os.unindent();
  return success();
}

LogicalResult TranslateModuleImpl::emitOperation(Operation &op) {
  if (auto amx = dyn_cast<AMXAsmOpInterface>(&op)) {
    amx.printAsm(os);
    return success();
  }
  if (isa<func::ReturnOp>(&op)) {
    os << "retq\n";
    return success();
  }
  return op.emitError() << "amx TranslateToAsm: unsupported op '"
                        << op.getName() << "'";
}

LogicalResult TranslateModuleImpl::emitFunc(func::FuncOp func,
                                            size_t funcIndex) {
  assert(func.getBody().hasOneBlock() &&
         "amx TranslateToAsm: multi-block functions not supported");
  if (failed(emitFuncTileCfg(func)))
    return failure();
  if (failed(emitFuncPrologue(func)))
    return failure();
  if (failed(emitBlock(func.getBody().front())))
    return failure();
  return emitFuncEpilogue(func, funcIndex);
}

LogicalResult TranslateModuleImpl::translate() {
  size_t funcIndex = 0;
  // Walk all nested regions to find func.func ops (they may be inside
  // x86.module or directly under builtin.module).
  LogicalResult result = success();
  module.walk([&](func::FuncOp func) {
    if (failed(result))
      return;
    if (failed(emitFunc(func, funcIndex++)))
      result = failure();
  });
  return result;
}

} // namespace

LogicalResult mlir::aster::amx::translateToAsm(Operation *op, raw_ostream &os) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return op->emitError() << "amx TranslateToAsm expects a ModuleOp";
  return TranslateModuleImpl(moduleOp, os).translate();
}
