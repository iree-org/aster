//===- IntEquivalenceClasses.h - Integer equivalence classes with members -===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_SUPPORT_INT_EQUIVALENCE_CLASSES_H
#define ASTER_SUPPORT_INT_EQUIVALENCE_CLASSES_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntEqClasses.h"
#include <cstdint>

namespace mlir::aster {
/// Integer equivalence classes with O(1) leader lookup and O(|class|) member
/// iteration.
class IntEquivalenceClasses {
public:
  IntEquivalenceClasses(int32_t numClasses) : eqClasses(numClasses) {
    assert(numClasses >= 0 && "number of classes must be non-negative");
    numElements = numClasses;
  }

  /// Check if the equivalence classes are compressed.
  bool isCompressed() const { return eqClasses.getNumClasses() != 0; }

  /// Join the classes of a and b.
  void join(int32_t a, int32_t b);

  /// Return the sorted list of members of the class containing x. Returns an
  /// empty ArrayRef if the class has only one member (x itself).
  ArrayRef<int32_t> getMembers(int32_t x) const {
    int32_t leader = getLeader(x);
    auto it = members.find(leader);
    if (it == members.end())
      return ArrayRef<int32_t>();
    return it->second;
  }
  /// Similar to getMembers(int32_t x) but always returns a non-empty ArrayRef.
  ArrayRef<int32_t> getMembers(int32_t x, int32_t &storage) const {
    ArrayRef<int32_t> members = getMembers(x);
    if (members.empty()) {
      storage = x;
      return ArrayRef<int32_t>(&storage, 1);
    }
    return members;
  }

  /// Get the leader of the class containing x.
  int32_t getLeader(int32_t x) const {
    return isCompressed() ? eqClasses[x] : eqClasses.findLeader(x);
  }
  int32_t operator[](int32_t x) const { return getLeader(x); }

  /// Compress the equivalence classes.
  void compress();

  /// Get the number of equivalence classes.
  int32_t getNumClasses() const { return eqClasses.getNumClasses(); }

  /// Get the number of elements.
  int32_t size() const { return numElements; }

  /// Print the equivalence classes.
  void print(raw_ostream &os) const;

private:
  int32_t numElements;
  llvm::IntEqClasses eqClasses;
  DenseMap<int32_t, SmallVector<int32_t, 4>> members;
};

} // namespace mlir::aster

#endif // ASTER_SUPPORT_INT_EQUIVALENCE_CLASSES_H
