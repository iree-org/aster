# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Emitter context (mutable state) and emission context (user-facing base)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from aster import ir

from .scope import ScopeStack
from .semantic import SemanticAnalyzer
from .table import EmitterTable
from .types import ASTType


@dataclass
class EmitterContext:
    """Mutable emission state threaded through all emitter callbacks."""

    module: ir.Module
    ctx: ir.Context
    ip: ir.InsertionPoint
    loc: ir.Location
    scope_stack: ScopeStack
    semantic: SemanticAnalyzer
    table: EmitterTable
    # Return type annotations for the current function, if any.
    return_types: list[ASTType] = field(default_factory=list)


# ---------------------------------------------------------------------------
# User-facing emission context (nestable).
# ---------------------------------------------------------------------------


class EmissionContext:
    """Base class for user-defined emission contexts.

    Subclass and override ``configure_*`` methods, then register via
    ``register_context('name', MyContext)``.
    """

    def configure_table(self, parent: EmitterTable) -> EmitterTable:
        """Return a new table inheriting from *parent* with overrides."""
        return EmitterTable(parent=parent)

    def configure_semantic(self, parent: SemanticAnalyzer) -> SemanticAnalyzer:
        """Return a semantic analyzer for this context."""
        return parent

    def configure_scope(self) -> dict[str, Any]:
        """Return scope configuration.

        Recognised keys:
          ``isolated`` (bool): if ``True``, locals from the parent scope
          are not visible inside this context.
        """
        return {"isolated": False}


# ---------------------------------------------------------------------------
# Context registry.
# ---------------------------------------------------------------------------

_context_registry: dict[str, type[EmissionContext]] = {}


def register_context(name: str, ctx_class: type[EmissionContext]) -> None:
    """Register *ctx_class* under *name* for use as ``@aster.<name>()``."""
    _context_registry[name] = ctx_class


def get_context_class(name: str) -> Optional[type[EmissionContext]]:
    """Look up a registered emission context by name."""
    return _context_registry.get(name)
