# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Two-tier name resolution: layered symbol table + local scope stack."""

from __future__ import annotations

from typing import Any, Optional


class SymbolTable:
    """Layered symbol table for functions, classes, globals.

    Each layer can shadow entries from its parent.  Lookups walk the
    chain upward until a match is found.
    """

    def __init__(self, parent: Optional[SymbolTable] = None):
        self._parent = parent
        self._symbols: dict[str, Any] = {}

    def define(self, name: str, value: Any) -> None:
        """Bind *name* in this layer (may shadow a parent entry)."""
        self._symbols[name] = value

    def lookup(self, name: str) -> Any:
        """Return the value for *name*, walking the parent chain.

        Raises ``KeyError`` if the name is not found anywhere.
        """
        if name in self._symbols:
            return self._symbols[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        raise KeyError(name)

    def contains(self, name: str) -> bool:
        """Return whether *name* is defined in this layer or any parent."""
        if name in self._symbols:
            return True
        if self._parent is not None:
            return self._parent.contains(name)
        return False

    def fork(self) -> SymbolTable:
        """Create a child layer that inherits from this table."""
        return SymbolTable(parent=self)


class Scope:
    """A single local-variable scope (function-level or explicit).

    Variable bindings are created by assignment or parameter binding.
    Lookups walk the optional parent chain so inner scopes can read
    enclosing locals.
    """

    def __init__(self, parent: Optional[Scope] = None):
        self._parent = parent
        self._locals: dict[str, Any] = {}

    def bind(self, name: str, value: Any) -> None:
        """Bind *name* in this scope."""
        self._locals[name] = value

    def lookup(self, name: str) -> Any:
        """Return the value for *name*, walking the parent chain.

        Raises ``KeyError`` if the name is not found.
        """
        if name in self._locals:
            return self._locals[name]
        if self._parent is not None:
            return self._parent.lookup(name)
        raise KeyError(name)

    def contains(self, name: str) -> bool:
        """Return whether *name* is bound in this scope or any parent."""
        if name in self._locals:
            return True
        if self._parent is not None:
            return self._parent.contains(name)
        return False


class ScopeStack:
    """Manages the scope stack and provides unified name resolution.

    Resolution order: current scope locals (walking the scope's parent
    chain) then the symbol table (walking its parent chain).
    """

    def __init__(self, symbols: SymbolTable):
        self._symbols = symbols
        self._scopes: list[Scope] = []

    @property
    def symbols(self) -> SymbolTable:
        return self._symbols

    @symbols.setter
    def symbols(self, value: SymbolTable) -> None:
        self._symbols = value

    def push_scope(self, parent: Optional[Scope] = None) -> Scope:
        """Push a new local scope.

        If *parent* is ``None`` the new scope chains to the current top
        scope (standard Python semantics).  Pass an explicit ``Scope()``
        with no parent to create an isolated scope.
        """
        if parent is None and self._scopes:
            parent = self._scopes[-1]
        scope = Scope(parent=parent)
        self._scopes.append(scope)
        return scope

    def push_isolated_scope(self) -> Scope:
        """Push a scope with no parent chain (isolated)."""
        scope = Scope(parent=None)
        self._scopes.append(scope)
        return scope

    def pop_scope(self) -> Scope:
        """Pop and return the top scope."""
        return self._scopes.pop()

    def current_scope(self) -> Scope:
        """Return the top scope."""
        return self._scopes[-1]

    def resolve(self, name: str) -> Any:
        """Resolve *name*: locals first, then symbols.

        Raises ``KeyError`` if not found in either.
        """
        if self._scopes:
            try:
                return self.current_scope().lookup(name)
            except KeyError:
                pass
        return self._symbols.lookup(name)

    def bind_local(self, name: str, value: Any) -> None:
        """Bind *name* in the current scope."""
        self.current_scope().bind(name, value)
