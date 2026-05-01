# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pluggable emitter table with parent-chain inheritance."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    from .context import EmitterContext
    from .types import ASTValue, ASTType

# ---------------------------------------------------------------------------
# Handler type aliases.
# ---------------------------------------------------------------------------
# All handlers receive an EmitterContext as the first argument.

BinOpEmitter = Callable[["EmitterContext", "ASTValue", "ASTValue"], "ASTValue"]
UnaryOpEmitter = Callable[["EmitterContext", "ASTValue"], "ASTValue"]
CmpOpEmitter = Callable[["EmitterContext", "ASTValue", "ASTValue"], "ASTValue"]
FunctionEmitter = Callable[
    ["EmitterContext", ast.FunctionDef, list["ASTType"], "ASTEmitterProtocol"],
    None,
]
ReturnEmitter = Callable[["EmitterContext", list["ASTValue"]], None]
CallEmitter = Callable[["EmitterContext", Any, list["ASTValue"]], "ASTValue"]
IfEmitter = Callable[
    [
        "EmitterContext",
        "ASTValue",
        list[ast.stmt],
        list[ast.stmt],
        "ASTEmitterProtocol",
    ],
    None,
]
ForEmitter = Callable[
    [
        "EmitterContext",
        str,
        "ASTValue",
        "ASTValue",
        "ASTValue",
        list[ast.stmt],
        "ASTEmitterProtocol",
    ],
    None,
]
ClassEmitter = Callable[
    ["EmitterContext", ast.ClassDef, "ASTEmitterProtocol"],
    None,
]
AttributeEmitter = Callable[["EmitterContext", Any, str], Any]
NameEmitter = Callable[["EmitterContext", Any], Any]
MethodCallEmitter = Callable[
    ["EmitterContext", Any, str, list[Any]], Optional["ASTValue"]
]
UnresolvedNameHandler = Callable[["EmitterContext", str], Any]


class ASTEmitterProtocol(Protocol):
    """Minimal protocol for the visitor so emitters can call back into it."""

    def visit(self, node: ast.AST) -> Any: ...

    def visit_stmts(self, stmts: list[ast.stmt]) -> None: ...

    def resolve_type_annotation(self, annotation: ast.AST) -> "ASTType": ...


class EmitterTable:
    """Registry of per-node-type emitter handlers.

    Lookups walk the parent chain: child overrides win, missing entries
    fall through to the parent.
    """

    def __init__(self, parent: Optional[EmitterTable] = None):
        self._parent = parent
        self._binops: dict[type[ast.operator], BinOpEmitter] = {}
        self._unaryops: dict[type[ast.unaryop], UnaryOpEmitter] = {}
        self._cmpops: dict[type[ast.cmpop], CmpOpEmitter] = {}
        self._calls: dict[Any, CallEmitter] = {}
        self._function_emitter: Optional[FunctionEmitter] = None
        self._return_emitter: Optional[ReturnEmitter] = None
        self._if_emitter: Optional[IfEmitter] = None
        self._for_emitter: Optional[ForEmitter] = None
        self._class_emitter: Optional[ClassEmitter] = None
        self._attribute_emitters: dict[type, AttributeEmitter] = {}
        self._name_emitters: dict[type, NameEmitter] = {}
        self._method_call_emitters: dict[type, MethodCallEmitter] = {}
        self._unresolved_name_handler: Optional[UnresolvedNameHandler] = None

    # -- registration ---------------------------------------------------------

    def register_binop(self, op: type[ast.operator], handler: BinOpEmitter) -> None:
        self._binops[op] = handler

    def register_unaryop(self, op: type[ast.unaryop], handler: UnaryOpEmitter) -> None:
        self._unaryops[op] = handler

    def register_cmpop(self, op: type[ast.cmpop], handler: CmpOpEmitter) -> None:
        self._cmpops[op] = handler

    def register_call(self, callee: Any, handler: CallEmitter) -> None:
        self._calls[callee] = handler

    def register_function_emitter(self, handler: FunctionEmitter) -> None:
        self._function_emitter = handler

    def register_return_emitter(self, handler: ReturnEmitter) -> None:
        self._return_emitter = handler

    def register_if_emitter(self, handler: IfEmitter) -> None:
        self._if_emitter = handler

    def register_for_emitter(self, handler: ForEmitter) -> None:
        self._for_emitter = handler

    def register_class_emitter(self, handler: ClassEmitter) -> None:
        self._class_emitter = handler

    def register_attribute_emitter(
        self, obj_type: type, handler: AttributeEmitter
    ) -> None:
        self._attribute_emitters[obj_type] = handler

    def register_name_emitter(self, obj_type: type, handler: NameEmitter) -> None:
        self._name_emitters[obj_type] = handler

    def register_method_call_emitter(
        self, obj_type: type, handler: MethodCallEmitter
    ) -> None:
        self._method_call_emitters[obj_type] = handler

    def register_unresolved_name_handler(self, handler: UnresolvedNameHandler) -> None:
        self._unresolved_name_handler = handler

    # -- lookup (walks parent chain) ------------------------------------------

    def get_binop_emitter(self, op: type[ast.operator]) -> BinOpEmitter:
        if op in self._binops:
            return self._binops[op]
        if self._parent is not None:
            return self._parent.get_binop_emitter(op)
        raise KeyError(f"no emitter registered for binary op {op.__name__}")

    def get_unaryop_emitter(self, op: type[ast.unaryop]) -> UnaryOpEmitter:
        if op in self._unaryops:
            return self._unaryops[op]
        if self._parent is not None:
            return self._parent.get_unaryop_emitter(op)
        raise KeyError(f"no emitter registered for unary op {op.__name__}")

    def get_cmpop_emitter(self, op: type[ast.cmpop]) -> CmpOpEmitter:
        if op in self._cmpops:
            return self._cmpops[op]
        if self._parent is not None:
            return self._parent.get_cmpop_emitter(op)
        raise KeyError(f"no emitter registered for compare op {op.__name__}")

    def get_call_emitter(self, callee: Any) -> Optional[CallEmitter]:
        if callee in self._calls:
            return self._calls[callee]
        # Fall back to type-based lookup for sentinel objects like _CallableRef.
        callee_type = type(callee)
        if callee_type in self._calls:
            return self._calls[callee_type]
        if self._parent is not None:
            return self._parent.get_call_emitter(callee)
        return None

    def get_function_emitter(self) -> FunctionEmitter:
        if self._function_emitter is not None:
            return self._function_emitter
        if self._parent is not None:
            return self._parent.get_function_emitter()
        raise KeyError("no function emitter registered")

    def get_return_emitter(self) -> ReturnEmitter:
        if self._return_emitter is not None:
            return self._return_emitter
        if self._parent is not None:
            return self._parent.get_return_emitter()
        raise KeyError("no return emitter registered")

    def get_if_emitter(self) -> IfEmitter:
        if self._if_emitter is not None:
            return self._if_emitter
        if self._parent is not None:
            return self._parent.get_if_emitter()
        raise KeyError("no if emitter registered")

    def get_for_emitter(self) -> ForEmitter:
        if self._for_emitter is not None:
            return self._for_emitter
        if self._parent is not None:
            return self._parent.get_for_emitter()
        raise KeyError("no for emitter registered")

    def get_class_emitter(self) -> ClassEmitter:
        if self._class_emitter is not None:
            return self._class_emitter
        if self._parent is not None:
            return self._parent.get_class_emitter()
        raise KeyError("no class emitter registered")

    def get_attribute_emitter(self, obj_type: type) -> Optional[AttributeEmitter]:
        if obj_type in self._attribute_emitters:
            return self._attribute_emitters[obj_type]
        if self._parent is not None:
            return self._parent.get_attribute_emitter(obj_type)
        return None

    def get_name_emitter(self, obj_type: type) -> Optional[NameEmitter]:
        if obj_type in self._name_emitters:
            return self._name_emitters[obj_type]
        if self._parent is not None:
            return self._parent.get_name_emitter(obj_type)
        return None

    def get_method_call_emitter(self, obj_type: type) -> Optional[MethodCallEmitter]:
        if obj_type in self._method_call_emitters:
            return self._method_call_emitters[obj_type]
        if self._parent is not None:
            return self._parent.get_method_call_emitter(obj_type)
        return None

    def get_unresolved_name_handler(self) -> Optional[UnresolvedNameHandler]:
        if self._unresolved_name_handler is not None:
            return self._unresolved_name_handler
        if self._parent is not None:
            return self._parent.get_unresolved_name_handler()
        return None
