cmake_minimum_required(VERSION 3.23)
project(aster VERSION 1.0)

# TODO: Remove this once we are on production
set(CMAKE_BUILD_TYPE DEBUG)

# Set CXX requirements
set(CMAKE_CXX_STANDARD 17)

################################################################################
# Options
################################################################################

option(ASTER_ENABLE_PYTHON "Enable building of the ASTER Python bindings" ON)

option(ASTER_EXTERNAL_LLVM "Enable building of the ASTER with an external LLVM" OFF)

################################################################################
# ASTER configuration
################################################################################

set(ASTER_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(ASTER_BINARY_DIR ${PROJECT_BINARY_DIR})

if(CMAKE_GENERATOR STREQUAL "Ninja" AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
  # Without this, installation fails because Ninja can't relink shared libraries
  # to change RPATHs from build tree to install tree on non-ELF platforms
  set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()


################################################################################
# Find LLVM & MLIR
################################################################################

add_subdirectory(llvm)

configure_llvm()

################################################################################
# Configure aster
################################################################################

set(ASTER_TOOLS_INSTALL_DIR ${PROJECT_BINARY_DIR}/bin)
set(ASTER_PYTHON_DIR ${ASTER_BINARY_DIR}/python_packages)

include_directories(${ASTER_SOURCE_DIR}/include)
include_directories(${ASTER_BINARY_DIR}/include)
add_subdirectory(tools/amdgcn-tblgen)

function(amdgcn_tablegen ofn)
  tablegen(Aster ${ARGV})
  set(TABLEGEN_OUTPUT
      ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE
  )
endfunction()

# Check if AMDGPU target is enabled in the LLVM build
if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(ASTER_ENABLE_TARGET 1)
else()
  set(ASTER_ENABLE_TARGET 0)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(ASTER_ENABLE_PYTHON)
  message(STATUS "ASTER Python bindings enabled")
  add_subdirectory(python)
endif()

add_subdirectory(test)
