# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Stub amd_comgr CONFIG package. Satisfies StinkyTofu's
# find_package(amd_comgr CONFIG) on platforms without a real ROCm install by
# providing an `amd_comgr` target backed by the no-op stub.c. The PUBLIC include
# dir exposes <amd_comgr/amd_comgr.h> to StinkyTofu's ComgrProbe.cpp.

if(TARGET amd_comgr)
  return()
endif()

# STATIC: the stub symbols are absorbed into the shared libstinkytofu; no separate
# dylib is produced or needs deploying.
add_library(amd_comgr STATIC "${CMAKE_CURRENT_LIST_DIR}/stub.c")
target_include_directories(amd_comgr PUBLIC "${CMAKE_CURRENT_LIST_DIR}")

set(amd_comgr_FOUND TRUE)
set(amd_comgr_VERSION "99.0.0")
