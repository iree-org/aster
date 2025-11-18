# Configure LLVM
macro(configure_llvm)
  if(ASTER_EXTERNAL_LLVM)
    find_package(MLIR REQUIRED CONFIG)
    find_package(LLD REQUIRED CONFIG)
  else()
    set(_MLIR_CMAKE_DIR "${ASTER_BINARY_DIR}/lib/cmake/mlir")
    find_package(MLIR REQUIRED CONFIG PATHS ${_MLIR_CMAKE_DIR})
  endif()

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  if(ASTER_EXTERNAL_LLVM)
    list(APPEND CMAKE_MODULE_PATH "${LLD_CMAKE_DIR}")
  endif()

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
  if(ASTER_EXTERNAL_LLVM)
    include_directories(${LLD_INCLUDE_DIRS})
  else()
    include_directories(${ASTER_BINARY_DIR}/llvm/llvm-project/tools/lld/include)
    include_directories(${ASTER_SOURCE_DIR}/llvm/llvm-project/lld/include)
  endif()

  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS} -Wno-deprecated-declarations)
endmacro()

if(ASTER_EXTERNAL_LLVM)
  return()
endif()


################################################################################
# Configure the bundled LLVM
################################################################################

# Configure build options
set(CMAKE_BUILD_TYPE Release)
set(LLVM_BUILD_TOOLS ON)
set(LLVM_ENABLE_PROJECTS "mlir;lld" CACHE STRING "Projects to build")
# We need to add host here because upstream is broken.
# TODO: Fix upstream.
set(LLVM_TARGETS_TO_BUILD "host;AMDGPU" CACHE STRING "Targets to build")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable assertions")
set(LLVM_CCACHE_BUILD ON CACHE BOOL "Use ccache")
set(CMAKE_CXX_FLAGS "-gmlt" CACHE STRING "CXX flags")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "Install utilities")
set(MLIR_ENABLE_BINDINGS_PYTHON ${ASTER_ENABLE_PYTHON} CACHE BOOL "Enable Python bindings")

# Add LLVM subdirectory
add_subdirectory(llvm-project/llvm EXCLUDE_FROM_ALL)
