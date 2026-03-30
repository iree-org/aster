# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""GEMM layout and swizzle constants for 16x16 MFMA + dwordx4 kernels."""

from aster.layout import Layout, Swizzle

# Cooperative 16x32 f16 tile layout for LDS (64-byte row stride).
# row = lane//4 (0..15), col_group = lane%4 (0..3, stride 16 bytes).
# Byte offset within 1024-byte tile: row*64 + col_group*16.
TILE_16x64 = Layout(sizes=(16, 4), strides=(64, 16))


def make_global_tile_layout(stride: int) -> Layout:
    """Tile load layout parametrized by global memory row stride in bytes.

    Same thread decomposition as TILE_16x64 (row=lane//4, col_group=lane%4), but uses
    the actual matrix row stride instead of the LDS tile stride (64).
    """
    return Layout(sizes=(16, 4), strides=(stride, 16))


# XOR swizzle for LDS bank conflict avoidance on 16x64-byte tiles.
# Extracts 3 row bits from position 6-8 of offset, XORs into bits 3-5.
LDS_SWIZZLE = Swizzle(bits=3, base=3, shift=3)

# MFMA A/B fragment layout within 16x64-byte LDS tile.
# group = lane//16 (0..3), row = lane%16 (0..15).
# Byte offset: group*8 + row*64. Each group reads 8 bytes (dwordx2).
MFMA_FRAG_IN_TILE = Layout(sizes=(4, 16), strides=(8, 64))


def make_mfma_c_layout(stride_c: int) -> Layout:
    """MFMA 16x16 C store layout parametrized by output row stride in bytes.

    Maps a flat coordinate (lane * 4 + agpr_i) to the byte offset in the output matrix.
    Delinearizes into (group, col, agpr_i):   group = coord // 64, col = (coord // 4) %
    16, agpr_i = coord % 4   row = group * 4 + agpr_i, byte_off = row * stride_c + col *
    4
    """
    return Layout(sizes=(4, 16, 4), strides=(4 * stride_c, 4, stride_c))
