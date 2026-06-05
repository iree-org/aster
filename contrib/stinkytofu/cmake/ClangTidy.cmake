# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# No-op shadow of the rocm-libraries cmake/modules/ClangTidy.cmake. StinkyTofu's
# root CMakeLists does include(ClangTidy) + add_clang_tidy_custom_target()
# unconditionally; the real module include()s CheckToolVersion.cmake and
# find_program()s clang-tidy under /opt/rocm. Defining every entry point here as
# a no-op keeps the include self-contained and avoids touching /opt/rocm.

macro(findAndCheckClangTidy)
endmacro()

function(setClangTidyVars)
endfunction()

function(clang_tidy_check)
endfunction()

function(add_clang_tidy_custom_target)
endfunction()
