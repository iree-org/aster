//===- AMDGCNInst.h - AMDGCN instructions -----------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN instructions.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_DIALECT_AMDGCN_IR_AMDGCNINST_H
#define AMDGCN_DIALECT_AMDGCN_IR_AMDGCNINST_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::aster {
namespace amdgcn {
class AMDGCNDialect;
/// Enumeration of AMDGCN opcodes.
enum class OpCode : uint64_t;

namespace detail {
struct InstAttrStorage;
}

/// Instruction metadata base class.
class InstMetadata {
public:
  virtual ~InstMetadata() = default;
  /// Get the opcode of the instruction.
  OpCode getOpCode() const { return opCode; }

  /// Equality operator.
  bool operator==(const InstMetadata &other) const {
    return opCode == other.opCode;
  }

  /// Hash the metadata.
  friend llvm::hash_code hash_value(const InstMetadata &md) {
    return llvm::hash_combine(static_cast<uint64_t>(md.getOpCode()));
  }

  /// Verify the instruction instance.
  virtual LogicalResult verify(Operation *op) const = 0;

protected:
  InstMetadata(OpCode opCode) : opCode(opCode) {}
  virtual void initialize(MLIRContext *ctx) {}
  friend struct amdgcn::detail::InstAttrStorage;

private:
  OpCode opCode;
};

/// CRTP helper class for defining instructions.
template <typename ConcreteTy>
class InstMD : public InstMetadata {
public:
  using Base = InstMD;
  InstMD() : InstMetadata(ConcreteTy::kOpCode) {}
  /// Classof method for LLVM-style RTTI.
  static bool classof(const InstMetadata *md) {
    return md->getOpCode() == ConcreteTy::kOpCode;
  }

  /// Verify the instruction instance.
  LogicalResult verify(Operation *op) const override {
    if (op->getName().getTypeID() !=
        TypeID::get<typename ConcreteTy::InstOp>()) {
      return op->emitError("unexpected operation type for instruction");
    }
    return static_cast<const ConcreteTy *>(this)->verifyImpl(
        cast<typename ConcreteTy::InstOp>(op));
  }
};
} // namespace amdgcn
} // namespace mlir::aster

#endif // AMDGCN_DIALECT_AMDGCN_IR_AMDGCNINST_H
