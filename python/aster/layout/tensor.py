# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aster.layout.algebra import Layout


@dataclass(frozen=True)
class Tensor:
    """View Tensor: pointer + dynamic offset + optional Layout."""

    ptr: Any
    offset: Any = None
    layout: Optional["Layout"] = None
