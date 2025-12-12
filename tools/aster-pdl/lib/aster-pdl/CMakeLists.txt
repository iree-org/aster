get_property(_capi_libraries GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)
add_mlir_aggregate(ASTERPDLCommon
  api.cpp
  SHARED
  EMBED_LIBS
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPISCF
    MLIRCAPIPDL
    MLIRCAPITransforms
    MLIRCAPIIR
    MLIRCAPIDebug
    MLIRCAPIInterfaces
    MLIRFuncInlinerExtension
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ASTERPDLCommon PRIVATE "-Wl,-exclude-libs,ALL")
else()
  if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
    message(STATUS "ASTER-PDL on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
  endif()
endif()
