# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# aster.layout -- layout algebra for GPU data mapping.

from aster.layout.int_tuple import (
    IntTuple,
    product,
    suffix_product,
    delinearize,
    linearize,
)
from aster.layout.algebra import Layout, Swizzle, make_layout
