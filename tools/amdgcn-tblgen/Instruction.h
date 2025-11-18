//===- Instruction.h ----------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Instruction and InstMap classes.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_TBLGEN_INSTRUCTION_H
#define AMDGCN_TBLGEN_INSTRUCTION_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"

#include <map>

namespace mlir::aster {
namespace amdgcn {
struct InstMap;
/// Represents an AMDGCN instruction with its variants and metadata.
struct Instruction {
  /// Default constructor.
  Instruction() = default;
  /// Returns the operation name for this instruction.
  StringRef getOpName() const { return opName; }

  /// Returns the instruction mnemonic.
  StringRef getOpMnemonic() const { return opMnemonic; }

  /// Returns the raw assembly string for this instruction.
  StringRef getAsmString() const { return asmString; }

  /// Returns all variants of this instruction.
  ArrayRef<const llvm::Record *> getInstVariants() const {
    return instVariants;
  }

  /// Creates a unique assembly identifier from an assembly string.
  static std::string getAsmIdentifier(StringRef asmString);

private:
  friend struct InstMap;
  /// Adds a new variant to this instruction.
  LogicalResult addInstVariant(StringRef asmStr, const llvm::Record &record,
                               llvm::DenseSet<StringRef> &types);
  /// The MLIR operation name (CamelCase version of mnemonic)
  std::string opName;
  /// The instruction mnemonic (first word of assembly string)
  StringRef opMnemonic;
  /// The raw assembly string from TableGen
  StringRef asmString;
  /// All TableGen records that represent variants of this instruction
  SmallVector<const llvm::Record *> instVariants;
};

/// A collection of AMDGCN instructions from TableGen records.
struct InstMap {
  /// Creates an InstMap from the TableGen records. Returns failure if the map
  /// couldn't be built.
  static FailureOr<InstMap>
  get(const llvm::RecordKeeper &records,
      function_ref<bool(const llvm::Record *)> filter = {});
  /// Returns the collection of all instruction records.
  const std::map<std::string, Instruction> &getInstructions() const {
    return instructions;
  }

private:
  /// Map of normalized assembly string to instruction data.
  /// Key: normalized assembly string (lowercase, no whitespace)
  /// Value: Instruction object containing all variants with that assembly
  /// pattern
  std::map<std::string, Instruction> instructions;
  /// Set of all operand types encountered during parsing.
  llvm::DenseSet<StringRef> usedTypes;
};
} // namespace amdgcn
} // namespace mlir::aster

#endif // AMDGCN_TBLGEN_INSTRUCTION_H
