# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shutil

import pytest

collect_ignore_glob = []

if not shutil.which("mlir-air-opt"):
    collect_ignore_glob.append("test/**")
