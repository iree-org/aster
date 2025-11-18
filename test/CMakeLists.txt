# Set LLVM_TOOLS_DIR to custom value if it was overridden by user
if(DEFINED _CUSTOM_LLVM_TOOLS_DIR AND _CUSTOM_LLVM_TOOLS_DIR)
  set(LLVM_TOOLS_DIR "${_CUSTOM_LLVM_TOOLS_DIR}")
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

if(NOT ASTER_EXTERNAL_LLVM)
  set(LLVM_DEPS
    FileCheck
    count
    not
  )
endif()

set(ASTER_TEST_DEPENDS
  aster-opt
  aster-translate
  ${LLVM_DEPS}
)

if(ASTER_ENABLE_PYTHON)
  list(APPEND ASTER_TEST_DEPENDS ASTERPythonModules)
  add_subdirectory(python/cdna)
endif()

add_lit_testsuite(check-aster "Running the aster regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ASTER_TEST_DEPENDS}
)

set_target_properties(check-aster PROPERTIES FOLDER "Tests")

add_lit_testsuites(ASTER ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ASTER_TEST_DEPENDS})
