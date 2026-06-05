# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds StinkyTofu's stinkytofu-opt from the vendored submodule. macOS has no
# ROCm, so comgr is satisfied by a no-op stub and a couple of small shims. No
# edits are made to the StinkyTofu source; every workaround lives in this dir.

# StinkyTofu is the shared/stinkytofu subtree of the sparse rocm-libraries submodule.
set(_STF_SRC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stinkytofu/shared/stinkytofu")
if(NOT EXISTS "${_STF_SRC}/CMakeLists.txt")
  message(FATAL_ERROR
    "StinkyTofu submodule not initialized at ${_STF_SRC}. "
    "See contrib/stinkytofu/README.md for the submodule bootstrap.")
endif()

# (1) Shadow the monorepo ClangTidy.cmake. StinkyTofu unconditionally does
#     include(ClangTidy) after appending ../../cmake/modules; prepend our no-op
#     module dir so our ClangTidy.cmake is found first (a prepend beats an append).
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# (2) comgr. On macOS / no-ROCm, satisfy find_package(amd_comgr CONFIG) with the
#     stub package under fake_comgr/. Flip OFF on Linux to require real ROCm comgr.
if(APPLE)
  set(_stf_stub_comgr_default ON)
else()
  # Linux: prefer a real ROCm comgr; fall back to the stub if absent so the
  # build still succeeds for tooling/analysis. Set -DSTINKYTOFU_USE_STUB_COMGR=OFF
  # to require real comgr.
  find_package(amd_comgr CONFIG QUIET)
  if(amd_comgr_FOUND)
    set(_stf_stub_comgr_default OFF)
  else()
    set(_stf_stub_comgr_default ON)
  endif()
endif()
option(STINKYTOFU_USE_STUB_COMGR
  "Use ASTER's no-op amd_comgr stub (OFF requires a real ROCm comgr)"
  ${_stf_stub_comgr_default})
if(STINKYTOFU_USE_STUB_COMGR)
  message(STATUS "stinkytofu: stub amd_comgr (no-ROCm path)")
  set(amd_comgr_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fake_comgr"
      CACHE PATH "amd_comgrConfig.cmake location" FORCE)
endif()

# (3) macOS shim: StinkyTofu's IntrinsicRegistry.cpp includes <linux/limits.h>.
#     Inject our compat dir ahead of the system search path for every StinkyTofu
#     target (directory-scoped, inherited by the subdir added below).
if(APPLE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/compat")
endif()

# (4) BUILD_SHARED_LIBS. StinkyTofu must build SHARED (its default): a static
#     stinkytofu propagates its PRIVATE amd_comgr dep into install(EXPORT) and
#     fails to generate, since the comgr stub is not in StinkyTofu's export set.
#     Force ON for the StinkyTofu subtree, then restore ASTER's value so ASTER's
#     own (statically linked) libraries are unaffected. ASTER's C++ targets are
#     all defined before this contrib, so the temporary ON does not reach them.
#     StinkyTofu's CXX_STANDARD 20 / -fno-rtti / -fno-exceptions are directory
#     scoped and do not leak to ASTER's C++17 parent targets.
if(DEFINED BUILD_SHARED_LIBS)
  set(_stf_saved_bsl "${BUILD_SHARED_LIBS}")
  set(_stf_bsl_was_defined TRUE)
else()
  set(_stf_bsl_was_defined FALSE)
endif()
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)

# Keep StinkyTofu's tests/python out of the build (rocisa is already gated by
# if(TARGET _rocisa), which ASTER never defines).
set(STINKYTOFU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(STINKYTOFU_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

# ASTER needs only stinkytofu-opt. EXCLUDE_FROM_ALL keeps StinkyTofu's own dev
# tooling (stinkytofu-check, intrinsic-compiler, test_loader, demo_auto_load) out
# of ASTER's default build -- that tooling belongs to the rocm-libraries repo and
# we do not duplicate it. CMAKE_SKIP_INSTALL_RULES suppresses StinkyTofu's install
# rules (which would otherwise pull the excluded tools back in via install() and
# dump StinkyTofu headers/cmake config into the venv); ASTER installs only
# stinkytofu-opt + libstinkytofu below. CMAKE_BUILD_WITH_INSTALL_RPATH FALSE gives
# the build-tree tool an rpath to the build-tree libstinkytofu (ASTER sets it ON
# project-wide).
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_SKIP_INSTALL_RULES TRUE)

add_subdirectory("${_STF_SRC}" "${CMAKE_CURRENT_BINARY_DIR}/stinkytofu" EXCLUDE_FROM_ALL)

set(CMAKE_SKIP_INSTALL_RULES FALSE)

# Restore ASTER's BUILD_SHARED_LIBS: put back its prior value, or clear the cache
# entry entirely if ASTER never set one (so we leave no permanent OFF behind).
if(_stf_bsl_was_defined)
  set(BUILD_SHARED_LIBS "${_stf_saved_bsl}" CACHE BOOL "Build shared libraries" FORCE)
else()
  unset(BUILD_SHARED_LIBS CACHE)
endif()

# Re-include just stinkytofu-opt (and, transitively, its link deps) in the default
# build, and point the installed tool at libstinkytofu in the install tree.
if(TARGET stinkytofu-opt)
  set_target_properties(stinkytofu-opt PROPERTIES
    EXCLUDE_FROM_ALL FALSE
    INSTALL_RPATH "@loader_path/../lib")
endif()

# Install only what ASTER needs: the tool and the shared library it links.
install(TARGETS stinkytofu-opt RUNTIME DESTINATION bin COMPONENT stinkytofu)
install(TARGETS stinkytofu LIBRARY DESTINATION lib COMPONENT stinkytofu)

# The lit suite DEPENDS on stinkytofu-opt; only register it if that target exists.
if(TARGET stinkytofu-opt)
  add_subdirectory(test)
endif()
