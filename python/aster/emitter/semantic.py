# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Semantic analysis protocol and default implementation."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Protocol, Union

from aster import ir
from aster.dialects import arith

from .types import ASTType, ASTValue, IndexType, ScalarType

if TYPE_CHECKING:
    from .context import EmitterContext


class SemanticAnalyzer(Protocol):
    """Interface for type inference, coercion, and validation.

    Implementations are swappable per-context so that different emission
    contexts can apply different type rules.
    """

    def infer_binop_type(
        self, op: type[ast.operator], lhs: ASTType, rhs: ASTType
    ) -> ASTType:
        """Infer the result type of a binary operation."""
        ...

    def coerce(
        self, ectx: EmitterContext, value: ASTValue, target: ASTType
    ) -> ASTValue:
        """Insert coercion ops to convert *value* to *target* type."""
        ...

    def check_return(self, declared: list[ASTType], actual: list[ASTType]) -> None:
        """Validate that returned types match the declared return types."""
        ...

    def materialize_constant(
        self, ectx: EmitterContext, value: Union[int, float], target: ASTType
    ) -> ASTValue:
        """Materialise a Python literal into an ``arith.constant``."""
        ...


class DefaultSemantic:
    """Basic integer/float promotion and coercion rules backed by arith ops."""

    # -- type inference -------------------------------------------------------

    def infer_binop_type(
        self, op: type[ast.operator], lhs: ASTType, rhs: ASTType
    ) -> ASTType:
        if isinstance(lhs, IndexType) and isinstance(rhs, IndexType):
            return IndexType()

        if not isinstance(lhs, ScalarType) or not isinstance(rhs, ScalarType):
            raise TypeError(f"binary op on unsupported types: {lhs}, {rhs}")

        # Float wins over int.
        if lhs.is_float or rhs.is_float:
            width = max(
                lhs.width if lhs.is_float else 0,
                rhs.width if rhs.is_float else 0,
            )
            return ScalarType(width=width, is_float=True)

        # Both integers -- widen to the larger.
        return ScalarType(
            width=max(lhs.width, rhs.width),
            is_float=False,
            is_signed=lhs.is_signed or rhs.is_signed,
        )

    # -- coercion -------------------------------------------------------------

    def coerce(
        self, ectx: EmitterContext, value: ASTValue, target: ASTType
    ) -> ASTValue:
        if value.ast_type == target:
            return value

        src = value.ast_type
        dst = target

        # Index -> integer.
        if isinstance(src, IndexType) and isinstance(dst, ScalarType):
            mlir_ty = dst.to_mlir_type(ectx.ctx)
            ir_val = arith.index_cast(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
            return ASTValue(ir_val, dst)

        # Integer -> index.
        if isinstance(src, ScalarType) and isinstance(dst, IndexType):
            mlir_ty = dst.to_mlir_type(ectx.ctx)
            ir_val = arith.index_cast(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
            return ASTValue(ir_val, dst)

        if not isinstance(src, ScalarType) or not isinstance(dst, ScalarType):
            raise TypeError(f"cannot coerce {src} to {dst}")

        # Int -> float.
        if not src.is_float and dst.is_float:
            mlir_ty = dst.to_mlir_type(ectx.ctx)
            if src.is_signed:
                ir_val = arith.sitofp(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
            else:
                ir_val = arith.uitofp(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
            return ASTValue(ir_val, dst)

        # Float -> wider float.
        if src.is_float and dst.is_float and src.width < dst.width:
            mlir_ty = dst.to_mlir_type(ectx.ctx)
            ir_val = arith.extf(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
            return ASTValue(ir_val, dst)

        # Int -> wider int.
        if not src.is_float and not dst.is_float and src.width < dst.width:
            mlir_ty = dst.to_mlir_type(ectx.ctx)
            if src.is_signed:
                ir_val = arith.extsi(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
            else:
                ir_val = arith.extui(mlir_ty, value.ir_value, loc=ectx.loc, ip=ectx.ip)
            return ASTValue(ir_val, dst)

        raise TypeError(f"cannot coerce {src} to {dst}")

    # -- return checking ------------------------------------------------------

    def check_return(self, declared: list[ASTType], actual: list[ASTType]) -> None:
        if len(declared) != len(actual):
            raise TypeError(
                f"expected {len(declared)} return values, got {len(actual)}"
            )
        for i, (d, a) in enumerate(zip(declared, actual)):
            if d != a:
                raise TypeError(f"return value {i}: expected {d}, got {a}")

    # -- constant materialisation ---------------------------------------------

    def materialize_constant(
        self,
        ectx: EmitterContext,
        value: Union[int, float],
        target: ASTType,
    ) -> ASTValue:
        mlir_ty = target.to_mlir_type(ectx.ctx)
        if isinstance(value, float) or (
            isinstance(target, ScalarType) and target.is_float
        ):
            attr = ir.FloatAttr.get(mlir_ty, float(value))
        elif isinstance(target, IndexType):
            attr = ir.IntegerAttr.get(mlir_ty, int(value))
        else:
            attr = ir.IntegerAttr.get(mlir_ty, int(value))
        ir_val = arith.ConstantOp(mlir_ty, attr, loc=ectx.loc, ip=ectx.ip).result
        return ASTValue(ir_val, target)
