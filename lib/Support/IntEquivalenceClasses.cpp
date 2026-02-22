//===- IntEquivalenceClasses.cpp - Integer equivalence classes ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Support/IntEquivalenceClasses.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <iterator>

using namespace mlir::aster;

void IntEquivalenceClasses::join(int32_t a, int32_t b) {
  assert(a >= 0 && a < numElements && "a is out of range");
  assert(b >= 0 && b < numElements && "b is out of range");
  int32_t leaderA = eqClasses.findLeader(a);
  int32_t leaderB = eqClasses.findLeader(b);
  if (leaderA == leaderB)
    return;

  llvm::ArrayRef<int32_t> membersA;
  llvm::ArrayRef<int32_t> membersB;
  std::array<int32_t, 1> singleA, singleB;
  // Get the members of the classes.
  if (auto itA = members.find(leaderA); itA != members.end()) {
    membersA = itA->second;
  } else {
    singleA[0] = leaderA;
    membersA = singleA;
  }
  if (auto itB = members.find(leaderB); itB != members.end()) {
    membersB = itB->second;
  } else {
    singleB[0] = leaderB;
    membersB = singleB;
  }

  // Join the classes.
  int32_t newLeader = eqClasses.join(a, b);

  // Remove the old classes if they are not single or are not the leader.
  if (newLeader != leaderA && membersA.size() > 1)
    members.erase(leaderA);
  if (newLeader != leaderB && membersB.size() > 1)
    members.erase(leaderB);

  // Merge the members of the classes.
  SmallVector<int32_t, 4> merged;
  std::set_union(membersA.begin(), membersA.end(), membersB.begin(),
                 membersB.end(), std::back_inserter(merged));

  assert(merged.size() > 1 && "merged must have at least 2 elements");
  members[newLeader] = std::move(merged);
}

void IntEquivalenceClasses::print(raw_ostream &os) const {
  int32_t tmp;
  os << "IntEquivalenceClasses {\n";
  for (int32_t i = 0; i < numElements; i++) {
    ArrayRef<int32_t> members = getMembers(i, tmp);
    // Skip classes that were already printed.
    if (members[0] < i)
      continue;
    os << "  " << llvm::interleaved_array(members) << "\n";
  }
  os << "}";
}

void IntEquivalenceClasses::compress() {
  eqClasses.compress();

  // Move the members to the compressed classes.
  SmallVector<int32_t> keys = llvm::to_vector(members.keys());
  for (int32_t key : keys) {
    int32_t eqClass = getLeader(key);
    if (eqClass == key)
      continue;

    // Update the members map with the class.
    members.insert({eqClass, std::move(members[key])});
    members.erase(key);
  }
}
