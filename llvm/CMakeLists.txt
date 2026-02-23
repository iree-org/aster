# Configure LLVM
macro(configure_llvm)
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLD REQUIRED CONFIG)

  # Verify LLVM commit matches the pin in llvm/LLVM_COMMIT.
  # The installed LLVM embeds its commit in VCSRevision.h.
  file(STRINGS "${ASTER_SOURCE_DIR}/llvm/LLVM_COMMIT" _ASTER_EXPECTED_LLVM_COMMIT LIMIT_COUNT 1)
  string(STRIP "${_ASTER_EXPECTED_LLVM_COMMIT}" _ASTER_EXPECTED_LLVM_COMMIT)
  set(_ASTER_VCS_HEADER "${LLVM_INCLUDE_DIRS}/llvm/Support/VCSRevision.h")
  if(EXISTS "${_ASTER_VCS_HEADER}")
    file(STRINGS "${_ASTER_VCS_HEADER}" _ASTER_LLVM_REVISION_LINE REGEX "LLVM_REVISION")
    string(REGEX MATCH "\"([0-9a-f]+)\"" _ASTER_LLVM_REVISION_MATCH "${_ASTER_LLVM_REVISION_LINE}")
    set(_ASTER_LLVM_REVISION "${CMAKE_MATCH_1}")
    if(NOT _ASTER_LLVM_REVISION STREQUAL _ASTER_EXPECTED_LLVM_COMMIT)
      message(FATAL_ERROR
        "LLVM commit mismatch!\n"
        "  Expected (llvm/LLVM_COMMIT): ${_ASTER_EXPECTED_LLVM_COMMIT}\n"
        "  Found    (VCSRevision.h):    ${_ASTER_LLVM_REVISION}\n"
        "Rebuild shared LLVM at the pinned commit. See README_devs.md.")
    endif()
  else()
    message(FATAL_ERROR
      "Cannot verify LLVM commit: ${_ASTER_VCS_HEADER} not found. "
      "Expected commit: ${_ASTER_EXPECTED_LLVM_COMMIT}")
  endif()

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLD_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${ASTER_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${ASTER_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${ASTER_BINARY_DIR})
  set(LLVM_BUILD_TOOLS ON)

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  include_directories(${LLD_INCLUDE_DIRS})

  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS} -Wno-deprecated-declarations)
endmacro()
