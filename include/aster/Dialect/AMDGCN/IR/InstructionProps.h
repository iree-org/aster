//===- InstructionProps.h - AMDGCN instruction properties --------*- C++
//-*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InstructionProps container for AMDGCN instruction
// properties.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_INSTRUCTIONPROPS_H
#define ASTER_DIALECT_AMDGCN_IR_INSTRUCTIONPROPS_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <bitset>

namespace mlir::aster::amdgcn {
/// Container for instruction properties backed by a bitset.
class InstructionProps {
public:
  InstructionProps(ArrayRef<InstProp> props) {
    for (InstProp prop : props)
      bits.set(static_cast<size_t>(prop), true);
  }

  /// Check if the instruction has the given property.
  bool hasProp(InstProp prop) const {
    return prop < InstProp::LastProp && bits[static_cast<int32_t>(prop)];
  }

  /// Check if the instruction has all of the given properties.
  bool hasProps(ArrayRef<InstProp> props) const {
    return llvm::all_of(props, [this](InstProp prop) { return hasProp(prop); });
  }

  /// Check if the instruction has any of the given properties.
  bool hasAnyProps(ArrayRef<InstProp> props) const {
    return llvm::any_of(props, [this](InstProp prop) { return hasProp(prop); });
  }

private:
  std::bitset<static_cast<size_t>(InstProp::LastProp)> bits;
};
} // namespace mlir::aster::amdgcn

#endif // ASTER_DIALECT_AMDGCN_IR_INSTRUCTIONPROPS_H
