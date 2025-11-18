//===- Utils.h - AMDGCN utils ------------------------------------*- C++
//-*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares AMDGCN utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_DIALECT_AMDGCN_IR_UTILS_H
#define AMDGCN_DIALECT_AMDGCN_IR_UTILS_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace mlir {
class OpBuilder;
namespace aster {

class RegisterTypeInterface;
namespace amdgcn {
class KernelOp;

/// Operand kinds.
enum class OperandKind : int8_t {
  Invalid = 0,
  SGPR,
  VGPR,
  AGPR,
  IntImm,
  FPImm,
};

/// Structure to hold kernel argument segment info.
struct KernelArgSegmentInfo {
  /// Default alignment if none is specified.
  static constexpr int32_t kDefaultAlignment = 8;
  int32_t size = 0;
  int32_t maxAlignment = kDefaultAlignment;
  SmallVector<int32_t> offsets;
  /// Get the kernel argument segment info for the given kernel.
  static KernelArgSegmentInfo get(KernelOp kernel);
};

/// Create an allocation for a given register type.
Value createAllocation(OpBuilder &builder, Location loc,
                       RegisterTypeInterface regTy);

/// Get the register kind of the given register type. Returns std::nullopt if
/// the type is not an AMD register type.
std::optional<RegisterKind> getRegKind(RegisterTypeInterface regTy);

/// Get the register type of given kind and size.
RegisterTypeInterface getRegisterType(MLIRContext *ctx, RegisterKind kind,
                                      int16_t size);

/// Helper to get a VGPR type.
inline RegisterTypeInterface getVGPR(MLIRContext *ctx, int16_t size = 1) {
  return getRegisterType(ctx, RegisterKind::VGPR, size);
}
/// Helper to get a SGPR type.
inline RegisterTypeInterface getSGPR(MLIRContext *ctx, int16_t size = 1) {
  return getRegisterType(ctx, RegisterKind::SGPR, size);
}
/// Helper to get an AGPR type.
inline RegisterTypeInterface getAGPR(MLIRContext *ctx, int16_t size = 1) {
  return getRegisterType(ctx, RegisterKind::AGPR, size);
}

/// Get the operand kind for the given type.
OperandKind getOperandKind(Type type);
/// Helper to check if the operand kind matches any of the given kinds.
inline bool isOperandOf(OperandKind value, ArrayRef<OperandKind> kinds) {
  return llvm::is_contained(kinds, value);
}

/// Check if the given type is an AMD register type.
bool isAMDReg(Type regTy);

/// Get the ISA version for the given target.
ISAVersion getIsaForTarget(Target target);

/// Check if an instruction opcode is valid for all the given ISA versions.
/// Returns true if the instruction is valid for ALL ISAs in the list.
bool isOpcodeValidForAllIsas(OpCode opcode, ArrayRef<ISAVersion> isas);

/// Check if the given type is an AMD register type or an Immedate type.
bool isAMDRegOrImm(Type type);

/// Check if the given type is an AMD register type of the given kind.
/// If numWords is > 0, also checks that the register range size matches.
bool isAMDRegOf(Type type, RegisterKind kind, int16_t numWords);

/// Helper to check if the type is a AGPR.
inline bool isAGPR(Type type, int16_t numWords) {
  return isAMDRegOf(type, RegisterKind::AGPR, numWords);
}
/// Helper to check if the type is a SGPR.
inline bool isSGPR(Type type, int16_t numWords) {
  return isAMDRegOf(type, RegisterKind::SGPR, numWords);
}
/// Helper to check if the type is a VGPR.
inline bool isVGPR(Type type, int16_t numWords) {
  return isAMDRegOf(type, RegisterKind::VGPR, numWords);
}

/// Helper to split a register range. Returns an empty ValueRange if the value
/// is not a register range.
ValueRange splitRange(OpBuilder &builder, Location loc, Value value);
} // namespace amdgcn
} // namespace aster
} // namespace mlir

#endif // AMDGCN_DIALECT_AMDGCN_IR_UTILS_H
