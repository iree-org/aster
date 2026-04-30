# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""AST-to-AST rewrite hooks.

Rewrites run after template substitution but before the emitter visits
the tree.  They are standard ``ast.NodeTransformer`` subclasses that
also expose a ``rewrite(tree) -> tree`` method.
"""

from __future__ import annotations

import ast
from typing import Protocol


class ASTRewrite(Protocol):
    """A single AST rewrite pass."""

    def rewrite(self, tree: ast.AST) -> ast.AST:
        """Transform *tree* and return the (possibly modified) result."""
        ...


class RewritePipeline:
    """Ordered pipeline of AST rewrites."""

    def __init__(self) -> None:
        self._rewrites: list[ASTRewrite] = []

    def register(self, rewrite: ASTRewrite) -> None:
        """Append *rewrite* to the pipeline."""
        self._rewrites.append(rewrite)

    def run(self, tree: ast.AST) -> ast.AST:
        """Run every registered rewrite in order."""
        for rw in self._rewrites:
            tree = rw.rewrite(tree)
        return tree

    def __len__(self) -> int:
        return len(self._rewrites)


# Global rewrite pipeline applied to every ``@jit`` function.
_global_pipeline = RewritePipeline()


def register_rewrite(rewrite: ASTRewrite) -> None:
    """Register a global AST rewrite pass."""
    _global_pipeline.register(rewrite)


def get_global_pipeline() -> RewritePipeline:
    """Return the global rewrite pipeline."""
    return _global_pipeline


def clear_global_pipeline() -> None:
    """Remove all rewrites from the global pipeline.

    Intended for use in tests to prevent registered rewrites from
    leaking across test cases.
    """
    _global_pipeline._rewrites.clear()
