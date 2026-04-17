# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_llvm_executable(aster-cpu-opt
  aster-cpu-opt.cpp
)
llvm_update_compile_flags(aster-cpu-opt)
target_link_libraries(aster-cpu-opt PRIVATE
  ASTERX86Dialect
  MLIRFuncDialect
  MLIROptLib
)
mlir_check_all_link_libraries(aster-cpu-opt)
install(TARGETS aster-cpu-opt)
