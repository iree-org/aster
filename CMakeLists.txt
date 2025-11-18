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

# Function adapted from:
# llvm-project/mlir/cmake/modules/AddMLIR.cmake
function(add_pdll_library target inputFile ofn)
  set(LLVM_TARGET_DEFINITIONS ${inputFile})

  _pdll_tablegen(MLIR_PDLL ${ofn} -x=mlir ${ARGN})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)

  # Get the current set of include paths for this pdll file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${inputFile})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${inputFile})
  endif()

  # Append the includes used for this file to the pdll_compilation_commands
  # file.
  file(APPEND ${CMAKE_BINARY_DIR}/pdll_compile_commands.yml
      "--- !FileInfo:\n"
      "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
      "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
  )

  # Convert the generated binary file to a C++ include file.
  set(_CONTENT "\
file(READ \${CMAKE_CURRENT_BINARY_DIR}/${ofn} filedata HEX)\n\
string(REGEX REPLACE \"([0-9a-f][0-9a-f])\" \"0x\\\\1,\" filedata \${filedata})\n\
file(WRITE \${CMAKE_CURRENT_BINARY_DIR}/${ofn}.inc \"const char ID[] = {\${filedata}};\\nconst int64_t SIZE_ID = sizeof(ID);\\n\")\n\
")
  file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen_${ofn}.cmake
    CONTENT "${_CONTENT}"
  )
  add_custom_target(${target}PDL DEPENDS ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/gen_${ofn}.cmake)
  add_custom_command(
    TARGET ${target}PDL POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/gen_${ofn}.cmake
    COMMENT "Generating C++ include file for ${ofn}"
    VERBATIM
  )
  # Generate the public target.
  add_public_tablegen_target(${target})
  add_dependencies(${target} ${target}PDL)
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
