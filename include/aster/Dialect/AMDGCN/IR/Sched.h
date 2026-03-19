//===- Sched.h - AMDGCN scheduling labeler ----------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AMDGCN scheduling labeler.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_AMDGCN_IR_SCHED_H
#define ASTER_DIALECT_AMDGCN_IR_SCHED_H

#include "aster/Dialect/AMDGCN/IR/AMDGCNInst.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir

namespace mlir::aster::amdgcn {

/// A pattern used to label scheduling operations.
struct SchedPattern {
  /// The label assigned when this pattern matches.
  int32_t label = 0;
  /// Benefit used to break ties between patterns.
  uint16_t benefit = 0;
  /// The TypeID of the operation type this pattern applies to.
  TypeID opTypeID = {};
  /// The instruction metadata this pattern applies to.
  const InstMetadata *instMetadata = nullptr;
  /// The instruction property this pattern applies to.
  std::optional<InstProp> instProp;

  /// Check if this pattern matches the given operation, instruction metadata.
  bool match(TypeID opId, const InstMetadata *metadata) const;
};

/// Labeler that assigns integer labels to operations based on patterns read
/// from a YAML configuration file.
///
/// The YAML file contains a list of items, each with:
///   - kind: one of [any, opcode, inst_prop, opname]
///   - label: non-negative integer
class SchedLabeler {
public:
  /// Read a SchedLabeler from a YAML file at the given path.
  static FailureOr<SchedLabeler> getFromYAML(StringRef path, MLIRContext *ctx);

  /// Try all patterns in order and return the label of the first matching
  /// pattern. Returns -1 if no pattern matches.
  int32_t getLabel(Operation *op) const;

  /// Return true if this labeler has no patterns.
  bool isTrivial() const { return patterns.empty(); }

  /// Return the name (path) of this labeler.
  StringRef getName() const { return name; }

  bool operator==(const SchedLabeler &other) const {
    return name == other.name;
  }

  void setName(StringRef newName) { name = newName.str(); }

private:
  /// The patterns used to label operations.
  llvm::SmallVector<SchedPattern> patterns;
  /// Map from (TypeID, InstMetadata*) to a (start, count) range into the
  /// patterns array, used to accelerate label lookup.
  llvm::DenseMap<std::pair<TypeID, const InstMetadata *>,
                 std::pair<int64_t, int64_t>>
      metadataToPatternPos;
  /// The name of this labeler.
  std::string name;
};

inline llvm::hash_code hash_value(const SchedLabeler &labeler) {
  return llvm::hash_value(labeler.getName());
}

/// Print the YAML path of the labeler as a quoted string.
AsmPrinter &operator<<(AsmPrinter &printer, const SchedLabeler &labeler);

} // namespace mlir::aster::amdgcn

namespace mlir {

/// Parse a SchedLabeler by reading a quoted YAML file path and loading it.
template <>
struct FieldParser<::mlir::aster::amdgcn::SchedLabeler,
                   ::mlir::aster::amdgcn::SchedLabeler> {
  static FailureOr<::mlir::aster::amdgcn::SchedLabeler>
  parse(AsmParser &parser);
};

} // namespace mlir

#endif // ASTER_DIALECT_AMDGCN_IR_SCHED_H
