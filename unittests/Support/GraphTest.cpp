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

} // namespace
