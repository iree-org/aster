# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Configs for GEMM kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GemmConfig:
    """Core GEMM problem geometry -- shared by all kernel variants."""

    m_wg: int
    n_wg: int
    m_waves: int
    n_waves: int
    m_tiles_wg: int
    n_tiles_wg: int
    k_tiles: int
    k: int
    in_elt_bytes: int = 2  # f16
    out_elt_bytes: int = 4  # f32

    @property
    def m_tiles(self) -> int:
        return self.m_tiles_wg // self.m_waves

    @property
    def n_tiles(self) -> int:
        return self.n_tiles_wg // self.n_waves

    @property
    def num_workgroups(self) -> int:
        return self.m_wg * self.n_wg

    @property
    def num_waves(self) -> int:
        return self.m_waves * self.n_waves

    @property
    def num_threads(self) -> int:
        return self.num_waves * 64

    @property
    def m_dim(self) -> int:
        return self.m_wg * self.m_tiles_wg * 16

    @property
    def n_dim(self) -> int:
        return self.n_wg * self.n_tiles_wg * 16

    @property
    def stride_a(self) -> int:
        return self.k * self.in_elt_bytes

    @property
    def stride_b(self) -> int:
        return self.k * self.in_elt_bytes

    @property
    def stride_c(self) -> int:
        return self.n_dim * self.out_elt_bytes

    @property
    def total_flops(self) -> int:
        return 2 * self.m_dim * self.n_dim * self.k
