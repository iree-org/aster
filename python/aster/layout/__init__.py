# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# aster.layout -- layout algebra for GPU data mapping.

from .int_tuple import IntTuple, product, prefix_product, delinearize, linearize
from .algebra import Layout, make_layout
from .codegen import Delinearize, Linearize, layout_to_ops
