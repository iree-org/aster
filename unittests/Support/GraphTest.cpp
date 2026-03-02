//===- GraphTest.cpp - Unit tests for Graph utilities ---------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Support/Graph.h"
#include "llvm/ADT/IntEqClasses.h"
#include "gtest/gtest.h"

using namespace mlir::aster;

namespace {

TEST(GraphTest, DirectedGraphBasic) {
  Graph g(true);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(0, 2);
  g.compress();

  EXPECT_TRUE(g.isDirected());
  EXPECT_EQ(g.sizeNodes(), 3);
  EXPECT_EQ(g.sizeEdges(), 3);
  EXPECT_TRUE(g.hasNode(0));
  EXPECT_TRUE(g.hasNode(1));
  EXPECT_TRUE(g.hasNode(2));
  EXPECT_FALSE(g.hasNode(3));
  EXPECT_TRUE(g.hasEdge(0, 1));
  EXPECT_TRUE(g.hasEdge(1, 2));
  EXPECT_TRUE(g.hasEdge(0, 2));
  EXPECT_FALSE(g.hasEdge(1, 0));
}

TEST(GraphTest, UndirectedGraphBasic) {
  Graph g(false);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.compress();

  EXPECT_FALSE(g.isDirected());
  EXPECT_EQ(g.sizeNodes(), 3);
  EXPECT_EQ(g.sizeEdges(), 2);
  EXPECT_TRUE(g.hasEdge(0, 1));
  EXPECT_TRUE(g.hasEdge(1, 0));
  EXPECT_TRUE(g.hasEdge(1, 2));
  EXPECT_TRUE(g.hasEdge(2, 1));
}

TEST(GraphTest, Compress) {
  Graph g(true);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  EXPECT_FALSE(g.isCompressed());
  g.compress();
  EXPECT_TRUE(g.isCompressed());
}

TEST(GraphTest, ComputeQuotientFailureWhenUncompressed) {
  Graph g(true);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.compress();

  llvm::IntEqClasses eqClasses(3);
  eqClasses.join(0, 2);
  // Not compressed - getNumClasses() is 0
  EXPECT_TRUE(failed(g.computeQuotient(eqClasses)));
}

TEST(GraphTest, ComputeQuotientSuccess) {
  Graph g(true);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.compress();

  llvm::IntEqClasses eqClasses(3);
  eqClasses.join(0, 2);
  eqClasses.compress();

  EXPECT_TRUE(succeeded(g.computeQuotient(eqClasses)));
  EXPECT_EQ(g.sizeNodes(), 2);
  EXPECT_EQ(g.sizeEdges(), 2);
  // Class 0 = {0, 2}, class 1 = {1}
  // Original edges: 0->1, 1->2 -> quotient: 0->1, 1->0
  EXPECT_TRUE(g.hasEdge(0, 1));
  EXPECT_TRUE(g.hasEdge(1, 0));
}

TEST(GraphTest, ComputeQuotientDiamondGraph) {
  // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
  Graph g(true);
  g.addEdge(0, 1);
  g.addEdge(0, 2);
  g.addEdge(1, 3);
  g.addEdge(2, 3);
  g.compress();

  // Merge nodes 1 and 2 into same class
  llvm::IntEqClasses eqClasses(4);
  eqClasses.join(1, 2);
  eqClasses.compress();

  EXPECT_TRUE(succeeded(g.computeQuotient(eqClasses)));
  EXPECT_EQ(g.sizeNodes(), 3);
  EXPECT_EQ(g.sizeEdges(), 2);
  // Class 0 = {0}, class 1 = {1, 2}, class 2 = {3}
  // 0->1, 0->2 -> 0->1; 1->3, 2->3 -> 1->2
  EXPECT_TRUE(g.hasEdge(0, 1));
  EXPECT_TRUE(g.hasEdge(1, 2));
}

TEST(GraphTest, ComputeQuotientUndirectedGraph) {
  // Chain: 0-1-2-3-4
  Graph g(false);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 3);
  g.addEdge(3, 4);
  g.compress();

  // Merge 0 with 2, and 1 with 3
  llvm::IntEqClasses eqClasses(5);
  eqClasses.join(0, 2);
  eqClasses.join(1, 3);
  eqClasses.compress();

  EXPECT_TRUE(succeeded(g.computeQuotient(eqClasses)));
  EXPECT_EQ(g.sizeNodes(), 3);
  // Class 0 = {0, 2}, class 1 = {1, 3}, class 2 = {4}
  // Edges: 0-1, 1-2, 2-3, 3-4 -> quotient: 0-1, 1-0, 1-2, 2-1
  EXPECT_EQ(g.sizeEdges(), 2);
  EXPECT_TRUE(g.hasEdge(0, 1));
  EXPECT_TRUE(g.hasEdge(1, 2));
}

TEST(GraphTest, ComputeQuotientAllNodesInOneClass) {
  Graph g(true);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.compress();

  llvm::IntEqClasses eqClasses(3);
  eqClasses.join(0, 1);
  eqClasses.join(1, 2);
  eqClasses.compress();

  EXPECT_TRUE(succeeded(g.computeQuotient(eqClasses)));
  EXPECT_EQ(g.sizeNodes(), 1);
  EXPECT_EQ(g.sizeEdges(), 1);
  // All edges collapse to self-loop 0->0
  EXPECT_TRUE(g.hasEdge(0, 0));
}

} // namespace
