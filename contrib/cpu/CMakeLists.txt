# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ASTER_ENABLE_CPU requires an x86_64-capable llvm-mc (and matching llvm-
# objdump) from the shared LLVM install. Verify at configure time so lit
# tests can use %llvm_mc / %llvm_objdump substitutions unconditionally.
set(_aster_llvm_mc "${LLVM_TOOLS_BINARY_DIR}/llvm-mc")
set(_aster_llvm_objdump "${LLVM_TOOLS_BINARY_DIR}/llvm-objdump")
if(NOT EXISTS "${_aster_llvm_mc}")
  message(FATAL_ERROR
    "ASTER_ENABLE_CPU=ON but ${_aster_llvm_mc} does not exist.\n"
    "The shared LLVM install is missing llvm-mc. Rebuild with:\n"
    "  ASTER_ENABLE_CPU=1 bash tools/setup.sh\n"
    "or manually:\n"
    "  LLVM_TARGETS_TO_BUILD=\"AMDGPU;X86\" bash tools/build-llvm.sh")
endif()

execute_process(
  COMMAND "${_aster_llvm_mc}" --version
  OUTPUT_VARIABLE _aster_llvm_mc_version
  ERROR_QUIET
  RESULT_VARIABLE _aster_llvm_mc_rc)
if(NOT _aster_llvm_mc_rc EQUAL 0)
  message(FATAL_ERROR
    "ASTER_ENABLE_CPU=ON: failed to run ${_aster_llvm_mc} --version")
endif()
if(NOT _aster_llvm_mc_version MATCHES "x86" AND
   NOT _aster_llvm_mc_version MATCHES "X86")
  message(FATAL_ERROR
    "ASTER_ENABLE_CPU=ON but ${_aster_llvm_mc} has no x86 target "
    "(registered targets lack x86-64).\n"
    "Rebuild shared LLVM with x86 support:\n"
    "  ASTER_ENABLE_CPU=1 bash tools/setup.sh\n"
    "or manually:\n"
    "  LLVM_TARGETS_TO_BUILD=\"AMDGPU;X86\" bash tools/build-llvm.sh")
endif()

set(ASTER_LLVM_MC "${_aster_llvm_mc}" CACHE INTERNAL
    "x86-capable llvm-mc from shared LLVM (contrib/cpu)")
set(ASTER_LLVM_OBJDUMP "${_aster_llvm_objdump}" CACHE INTERNAL
    "x86-capable llvm-objdump from shared LLVM (contrib/cpu)")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
