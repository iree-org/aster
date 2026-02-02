find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

include(AddMLIRPython)

set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN "ON")

# Specifies that all MLIR packages are co-located under npcomp.
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=aster.")

set(_PYTHON_BUILD_PREFIX "${POSEIDON_BINARY_DIR}/python_packages/")
set(_PYTHON_INSTALL_PREFIX "python_packages/aster")

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)

declare_mlir_python_sources(ASTERPythonSources)
declare_mlir_python_sources(ASTERPythonExtensions)

declare_mlir_python_sources(ASTERPythonSources.core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aster"
  ADD_TO_PARENT ASTERPythonSources
  SOURCES
    __init__.py
    utils.py
    pass_pipelines.py
    logging.py
    compilation.py
    execution.py
    testing.py
)

declare_mlir_python_extension(ASTERPythonExtensions._amdgcn
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib"
  MODULE_NAME _amdgcn
  ADD_TO_PARENT ASTERPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    amdgcn.cpp
  EMBED_CAPI_LINK_LIBS
    ASTER
)

declare_mlir_python_extension(ASTERPythonExtensions._site_initialize_0
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib"
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT ASTERPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    site_init.cpp
  EMBED_CAPI_LINK_LIBS
    ASTER
)

################################################################################
# Configure HIP (optional)
################################################################################
find_package(hip QUIET)

if(hip_FOUND)
  message(STATUS "HIP found: HIP support will be enabled")
  set(HAS_HIP_SUPPORT ON)
else()
  message(STATUS "HIP not found: building with stub implementations")
  set(HAS_HIP_SUPPORT OFF)
endif()

################################################################################
# Runtime Module
################################################################################
declare_mlir_python_extension(ASTERPythonExtensions._runtime_module
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib"
  MODULE_NAME _runtime_module
  ADD_TO_PARENT ASTERPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    RuntimeModule.cpp
)

# Set RPATH for runtime module
set_property(TARGET ASTERPythonExtensions._runtime_module PROPERTY INSTALL_RPATH_USE_LINK_PATH ON)

if(HAS_HIP_SUPPORT)
  target_compile_definitions(ASTERPythonExtensions._runtime_module INTERFACE HAS_HIP_SUPPORT)
  target_link_libraries(ASTERPythonExtensions._runtime_module INTERFACE hip::host hip::amdhip64)
else()
  # No HIP libraries to link - stubs will be used
endif()

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ASTERPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aster"
  TD_FILE dialects/AMDGCNOps.td
  SOURCES
    dialects/amdgcn.py
    dialects/api.py
  DIALECT_NAME amdgcn
  GEN_ENUM_BINDINGS_TD_FILE
    dialects/AMDGCNOps.td
)

################################################################################
# Generate the instruction bindings
################################################################################
set(AMDGCN_TD aster/dialects/AMDGCNOps.td)
set(LLVM_TARGET_DEFINITIONS ${AMDGCN_TD})
get_filename_component(relative_td_directory "${AMDGCN_TD}" DIRECTORY)
set(output_filename "${relative_td_directory}/_amdgcn_inst_gen.py")
amdgcn_tablegen("${output_filename}" -gen-inst-bindings --inst-py-import="aster.dialects.amdgcn")
add_public_tablegen_target(AMDGCNInstBindingsGen)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${output_filename}"
  DESTINATION "${_PYTHON_INSTALL_PREFIX}/dialects"
  COMPONENT ASTERPythonModules
)
add_custom_command(
  TARGET AMDGCNInstBindingsGen POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_BINARY_DIR}/${output_filename}"
    "${ASTER_BINARY_DIR}/python_packages/aster/dialects/_amdgcn_inst_gen.py"
)
add_dependencies(ASTER AMDGCNInstBindingsGen)
################################################################################

set(_SOURCE_COMPONENTS
  ASTERPythonExtensions
  ASTERPythonSources

  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonExtension.Core.type_stub_gen
  # DO NOT register all indiscriminately.
  # MLIRPythonExtension.RegisterEverything
)

add_mlir_python_modules(ASTERPythonModules
  ROOT_PREFIX "${ASTER_BINARY_DIR}/python_packages/aster"
  INSTALL_PREFIX "python_packages/aster"
  DECLARED_SOURCES
  ${_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS
    ASTER
    AsterExportASM
    LLVMSupport
)

# Install libASTER.dylib/.so alongside Python modules so they can find it
# This uses the ASTER target created in tools/aster-shlib
install(TARGETS ASTER
  LIBRARY
    DESTINATION python_packages/aster/_mlir_libs
    COMPONENT ASTERPythonModules
  RUNTIME
    DESTINATION python_packages/aster/_mlir_libs
    COMPONENT ASTERPythonModules
)

nanobind_add_stub(
  _aster_stub
  MODULE aster._mlir_libs._amdgcn
  INCLUDE_PRIVATE
  OUTPUT ../python_packages/aster/_mlir_libs/_amdgcn.pyi
  PYTHON_PATH "${CMAKE_BINARY_DIR}/python_packages"
  DEPENDS ASTERPythonModules
)

# Copy integration_test directory to build directory
file(GLOB_RECURSE integration_test_files
  "${CMAKE_SOURCE_DIR}/integration_test/*.py"
)
foreach(file ${integration_test_files})
  file(RELATIVE_PATH rel_file "${CMAKE_SOURCE_DIR}/integration_test" "${file}")
  configure_file(
    "${file}"
    "${CMAKE_BINARY_DIR}/integration_test/${rel_file}"
    COPYONLY
  )
endforeach()

# Copy mlir_kernels/test directory to build directory
file(GLOB_RECURSE mlir_kernels_test_files
  "${CMAKE_SOURCE_DIR}/mlir_kernels/test/*.py"
)
foreach(file ${mlir_kernels_test_files})
  file(RELATIVE_PATH rel_file "${CMAKE_SOURCE_DIR}/mlir_kernels/test" "${file}")
  configure_file(
    "${file}"
    "${CMAKE_BINARY_DIR}/mlir_kernels/test/${rel_file}"
    COPYONLY
  )
endforeach()

add_custom_target(aster DEPENDS ASTERPythonModules)
