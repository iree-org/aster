# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_llvm_executable(aster-cpu-translate
  aster-cpu-translate.cpp
)
llvm_update_compile_flags(aster-cpu-translate)
target_link_libraries(aster-cpu-translate PRIVATE
  ASTERX86Registration
  MLIRTranslateLib
)
mlir_check_all_link_libraries(aster-cpu-translate)
install(TARGETS aster-cpu-translate)
