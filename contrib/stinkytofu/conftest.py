# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# StinkyTofu vendored sources ship upstream pytest suites under third_party/.
# ASTER only builds stinkytofu-opt (STINKYTOFU_BUILD_PYTHON=OFF); skip them.
collect_ignore_glob = ["third_party/**"]
