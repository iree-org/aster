# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Contextual AST-to-MLIR emitter framework.

Public API
----------
Decorators and helpers:
    jit, escape, reflect, template, instantiate

Context registration:
    register_context, EmissionContext

AST rewrite registration:
    register_rewrite, clear_global_pipeline

Emitter table:
    EmitterTable, default_table

Types:
    ASTType, ScalarType, IndexType, ASTValue, Type,
    i1, i8, i16, i32, i64, f16, f32, f64, index

Semantic analysis:
    SemanticAnalyzer, DefaultSemantic
"""

from .context import EmissionContext, register_context
from .core import JITFunction, UnsupportedConstruct, default_table, jit
from .emitc_emitter import EmitCContext, EmitCSemantic
from .escape import escape, reflect
from .pattern_emitter import PatternContext, pattern
from .rewrite import clear_global_pipeline, register_rewrite
from .semantic import DefaultSemantic, SemanticAnalyzer
from .table import EmitterTable
from .template import TemplateFunction, instantiate, template
from .types import (
    ASTType,
    ASTValue,
    IndexType,
    ScalarType,
    Type,
    f16,
    f32,
    f64,
    i1,
    i8,
    i16,
    i32,
    i64,
    index,
)

__all__ = [
    "ASTType",
    "ASTValue",
    "DefaultSemantic",
    "EmissionContext",
    "EmitterTable",
    "IndexType",
    "JITFunction",
    "ScalarType",
    "SemanticAnalyzer",
    "TemplateFunction",
    "Type",
    "UnsupportedConstruct",
    "EmitCContext",
    "EmitCSemantic",
    "PatternContext",
    "clear_global_pipeline",
    "default_table",
    "escape",
    "f16",
    "f32",
    "f64",
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "index",
    "instantiate",
    "jit",
    "pattern",
    "reflect",
    "register_context",
    "register_rewrite",
    "template",
]
