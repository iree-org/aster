set(amdgcn_libs
  ASTERInit
  AsterAPI
)

get_property(_capi_libraries GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)
add_mlir_aggregate(ASTER
  SHARED
  EMBED_LIBS
    ${_capi_libraries}
    ${amdgcn_libs}
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ASTER PRIVATE "-Wl,-exclude-libs,ALL")
else()
  if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
    message(STATUS "ASTER on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
  endif()
endif()
