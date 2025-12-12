add_subdirectory(lib/aster-pdl)

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
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=aster.pdl")

set(_PYTHON_BUILD_PREFIX "${ASTER_BINARY_DIR}/python_packages/")
set(_PYTHON_INSTALL_PREFIX "python_packages/aster/pdl")

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)

declare_mlir_python_sources(ASTERPDLPythonSources)
declare_mlir_python_sources(ASTERPDLPythonExtensions)

declare_mlir_python_sources(ASTERPDLPythonSources.core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aster/pdl"
  ADD_TO_PARENT ASTERPDLPythonSources
  SOURCES
    __init__.py
    pdll.py
    utils.py
)

declare_mlir_python_extension(ASTERPDLPythonExtensions._pdl
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib"
  MODULE_NAME _pdl
  ADD_TO_PARENT ASTERPDLPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    pdl.cpp
  EMBED_CAPI_LINK_LIBS
    ASTERPDLCommon
)

declare_mlir_python_extension(ASTERPDLPythonExtensions._site_initialize_0
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib"
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT ASTERPDLPythonExtensions
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    site_init.cpp
  EMBED_CAPI_LINK_LIBS
    ASTERPDLCommon
)

################################################################################

set(_SOURCE_COMPONENTS
  ASTERPDLPythonExtensions
  ASTERPDLPythonSources

  MLIRPythonSources.Core
  MLIRPythonExtension.Core.type_stub_gen
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.pdl
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.scf
)

add_mlir_python_modules(ASTERPDLPythonModules
  ROOT_PREFIX "${ASTER_BINARY_DIR}/python_packages/aster/pdl"
  INSTALL_PREFIX "python_packages/aster"
  DECLARED_SOURCES
  ${_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS
    ASTERPDLCommon
    LLVMSupport
)

# Install libASTER.dylib/.so alongside Python modules so they can find it
# This uses the ASTER target created in tools/aster-shlib
install(TARGETS ASTERPDLCommon
  LIBRARY
    DESTINATION python_packages/aster/pdl/_mlir_libs
    COMPONENT ASTERPDLPythonModules
  RUNTIME
    DESTINATION python_packages/aster/pdl/_mlir_libs
    COMPONENT ASTERPDLPythonModules
)

nanobind_add_stub(
  _aster_pdl_stub
  MODULE aster.pdl._mlir_libs._pdl
  INCLUDE_PRIVATE
  OUTPUT ../python_packages/aster/pdl/_mlir_libs/_aster.pyi
  PYTHON_PATH "${CMAKE_BINARY_DIR}/python_packages"
  DEPENDS ASTERPDLPythonModules
)
