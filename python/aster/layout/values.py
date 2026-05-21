# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Layout-keyed bundles of in-flight SSA values and sync tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aster.layout.algebra import Layout, flat_index


@dataclass(frozen=True, slots=True)
class LayoutValues:
    """SSA payloads and/or tokens indexed by ``value_layout`` coordinates."""

    value_layout: Layout
    payloads: tuple[Any, ...] | None = None
    tokens: tuple[Any, ...] | None = None

    @classmethod
    def from_flat(
        cls,
        value_layout: Layout,
        *,
        payloads: tuple[Any | None, ...] | None = None,
        tokens: tuple[Any, ...] | None = None,
    ) -> "LayoutValues":
        if payloads is not None:
            assert len(payloads) == value_layout.size(), (
                f"payloads length {len(payloads)} != layout size {value_layout.size()}"
            )
            payloads = None if all(p is None for p in payloads) else tuple(payloads)
        if tokens is not None:
            assert len(tokens) == value_layout.size(), (
                f"tokens length {len(tokens)} != layout size {value_layout.size()}"
            )
        return cls(value_layout, payloads=payloads, tokens=tokens)

    @property
    def has_payloads(self) -> bool:
        return self.payloads is not None

    @property
    def has_tokens(self) -> bool:
        return self.tokens is not None

    def _flat_index(self, coord: int | tuple[int, ...]) -> int:
        if isinstance(coord, int):
            coord = (coord,)
        return flat_index(coord, self.value_layout)

    def _resolve_coord(self, coord: int | tuple[int, ...]) -> int | tuple[int, ...]:
        if isinstance(coord, tuple) and len(coord) == 1 and isinstance(coord[0], tuple):
            return coord[0]
        return coord

    def data_at(self, *coord: int | tuple[int, ...]) -> Any:
        assert self.payloads is not None, "LayoutValues has no payloads"
        idx = self._flat_index(self._resolve_coord(coord))
        value = self.payloads[idx]
        assert value is not None, f"no payload at layout coord {coord!r}"
        return value

    def token_at(self, *coord: int | tuple[int, ...]) -> Any:
        assert self.tokens is not None, "LayoutValues has no tokens"
        idx = self._flat_index(self._resolve_coord(coord))
        return self.tokens[idx]

    def __getitem__(self, key: int | tuple[int, ...]) -> Any:
        if isinstance(key, tuple):
            return self.data_at(*key)
        return self.data_at(key)

    def token_values(self) -> list[Any]:
        assert self.tokens is not None
        return list(self.tokens)

    def __iter__(self):
        """Iterate per-coord entries:
        - payloads + tokens -> yields ``(payload, tok)`` tuples
        - payloads only     -> yields ``payload``
        - tokens only       -> yields ``tok``
        """
        n = self.value_layout.size()
        for i in range(n):
            payload = self.payloads[i] if self.payloads is not None else None
            token = self.tokens[i] if self.tokens is not None else None
            if self.has_payloads and self.has_tokens:
                yield (payload, token)
            elif self.has_payloads:
                yield payload
            else:
                assert self.has_tokens, "LayoutValues has neither payloads nor tokens"
                yield token

    def __len__(self) -> int:
        return self.value_layout.size()


# Deprecated alias; prefer LayoutValues.
Result = LayoutValues
