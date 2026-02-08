//===- SSAMap.h - SSA value to ID mapping -----------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Maps SSA values to unique numeric IDs and tracks the operations and blocks
// that define values present in the map.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_IR_SSAMAP_H
#define ASTER_IR_SSAMAP_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace mlir::aster {

/// Maps SSA values to unique numeric IDs and tracks the operations and blocks
/// that define values present in the map.
class SSAMap {
public:
  /// Insert IDs for all results of the operation. Adds the operation to the
  /// tracked set if any results are inserted.
  void insertOp(Operation *op);

  /// Insert IDs for all block arguments. Adds the block to the tracked set if
  /// any arguments are inserted.
  void insertBlock(Block *block);

  /// Get the ID for a value, or add it if not present. If the value needs to be
  /// added, invokes insertOp or insertBlock as appropriate.
  int64_t getOrAddValue(Value value);

  /// Populate the map by walking the operation and all nested regions, calling
  /// insertOp and insertBlock for each operation and block encountered.
  void populateMap(Operation *op);

  /// Print information about all blocks and operations in the map.
  void printMapMembers(llvm::raw_ostream &os) const;

  /// Get IDs for the given values, appending to the output vector.
  template <typename RangeTy>
  void getIds(RangeTy &&values, llvm::SmallVectorImpl<int64_t> &out) const {
    out.reserve(out.size() + values.size());
    for (Value value : values) {
      auto it = valueToId.find(value);
      out.push_back(it != valueToId.end() ? it->second : -1);
    }
  }
  template <typename RangeTy>
  void getIds(RangeTy &&values,
              llvm::SmallVectorImpl<std::pair<Value, int64_t>> &out) const {
    out.reserve(out.size() + values.size());
    for (Value value : values) {
      auto it = valueToId.find(value);
      out.push_back({value, it != valueToId.end() ? it->second : -1});
    }
  }

  /// Get IDs for the given values, returning a new vector.
  template <typename RangeTy>
  llvm::SmallVector<int64_t> getIds(RangeTy &&values) const {
    llvm::SmallVector<int64_t> result;
    getIds(std::forward<RangeTy>(values), result);
    return result;
  }

  /// Look up the ID for a value. Returns -1 if not found.
  int64_t lookup(Value value) const;

  /// Check if a value is in the map.
  bool contains(Value value) const;

private:
  llvm::DenseMap<Value, int64_t> valueToId;
  llvm::SetVector<llvm::PointerUnion<Operation *, Block *>>
      irElementsWithValues;
};

} // namespace mlir::aster

#endif // ASTER_IR_SSAMAP_H
