# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Frontend type descriptors and value wrappers for the AST emitter."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from aster import ir


class ASTType(ABC):
    """Base class for frontend type descriptors.

    Each subclass knows how to produce the corresponding MLIR type.
    """

    @abstractmethod
    def to_mlir_type(self, ctx: ir.Context) -> ir.Type:
        """Return the MLIR type corresponding to this frontend type."""
        ...


@dataclass(frozen=True)
class ScalarType(ASTType):
    """Integer or floating-point scalar type."""

    width: int
    is_float: bool = False
    is_signed: bool = True

    def to_mlir_type(self, ctx: ir.Context) -> ir.Type:
        if self.is_float:
            if self.width == 16:
                return ir.F16Type.get(ctx)
            if self.width == 32:
                return ir.F32Type.get(ctx)
            if self.width == 64:
                return ir.F64Type.get(ctx)
            raise ValueError(f"unsupported float width: {self.width}")
        return ir.IntegerType.get_signless(self.width, ctx)


@dataclass(frozen=True)
class IndexType(ASTType):
    """MLIR index type."""

    def to_mlir_type(self, ctx: ir.Context) -> ir.Type:
        return ir.IndexType.get(ctx)


# Convenient type aliases used in annotations and template parameters.
i1 = ScalarType(width=1)
i8 = ScalarType(width=8)
i16 = ScalarType(width=16)
i32 = ScalarType(width=32)
i64 = ScalarType(width=64)
f16 = ScalarType(width=16, is_float=True)
f32 = ScalarType(width=32, is_float=True)
f64 = ScalarType(width=64, is_float=True)
index = IndexType()


@dataclass(frozen=True)
class ASTValue:
    """A value tracked by the emitter, pairing an SSA value with its type."""

    ir_value: ir.Value
    ast_type: ASTType


# Sentinel used in template parameter declarations.
class Type:
    """Sentinel indicating a template parameter that accepts an ASTType."""


class TemplateParamKind(Enum):
    """Kinds of template parameters."""

    TYPE = "Type"
    VALUE = "value"
    CALLABLE = "Callable"


@dataclass(frozen=True)
class TemplateParam:
    """Declaration of a single template parameter."""

    name: str
    kind: TemplateParamKind

    @staticmethod
    def from_decl(name: str, decl: Any) -> TemplateParam:
        """Infer the parameter kind from the declaration value."""
        if decl is Type:
            return TemplateParam(name, TemplateParamKind.TYPE)
        # typing.Callable or any callable object that is not itself a type
        # (e.g. a function, lambda, or functools.partial) → CALLABLE param.
        # Plain types like int/float/str are excluded because they are used
        # as value-param type hints (e.g. @template(N=int)).
        if decl is typing.Callable or (callable(decl) and not isinstance(decl, type)):
            return TemplateParam(name, TemplateParamKind.CALLABLE)
        # Anything else (int, float, str, a type class, etc.) is a
        # compile-time value parameter.
        return TemplateParam(name, TemplateParamKind.VALUE)
