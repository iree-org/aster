# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ASTER_AMX_TEST_DEPENDS
  aster-cpu-opt
  aster-cpu-translate
)

add_lit_testsuite(check-aster-amx
  "Running ASTER contrib/cpu (AMX) lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ASTER_AMX_TEST_DEPENDS}
)
set_target_properties(check-aster-amx PROPERTIES FOLDER "Tests")

add_lit_testsuites(ASTER_AMX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${ASTER_AMX_TEST_DEPENDS}
)
