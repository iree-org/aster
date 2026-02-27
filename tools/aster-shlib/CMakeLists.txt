set(amdgcn_libs
  ASTERInit
  AsterAPI
)

# Need finer granularity.
# get_property(_capi_libraries GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)
set(_capi_libraries
  MLIRCAPIIR
  MLIRCAPITransforms
  MLIRCAPIDebug
  MLIRCAPIInterfaces
)

# add_mlir_library() for in-tree ASTER libs does not pass ENABLE_AGGREGATION,
# so MLIR_AGGREGATE_OBJECTS is not set automatically. The obj.${lib} targets
# still exist (created by the legacy NEEDS_OBJECT_LIB path in AddMLIR.cmake).
# Manually set the two properties that add_mlir_aggregate() reads so it can
# embed the ASTER object code into the shared library.
foreach(_lib ${amdgcn_libs})
  if(NOT TARGET obj.${_lib})
    message(FATAL_ERROR "Expected obj.${_lib} for aggregation but target not found")
  endif()
  get_target_property(_link_libs ${_lib} LINK_LIBRARIES)
  get_mlir_filtered_link_libraries(_filtered_libs ${_link_libs})
  set_target_properties(${_lib} PROPERTIES
    MLIR_AGGREGATE_OBJECTS "$<TARGET_OBJECTS:obj.${_lib}>"
    MLIR_AGGREGATE_DEPS    "${_filtered_libs}"
  )
endforeach()

add_mlir_aggregate(ASTER
  SHARED
  EMBED_LIBS
    ${_capi_libraries}
    ${amdgcn_libs}
  PUBLIC_LIBS
    MLIRParser
    MLIRAsmParser
)
# Place ASTER.dylib/.so alongside the Python extensions in the build tree so
# that @loader_path RPATH on the extensions resolves at build time (stubgen
# runs before ninja install).
set_target_properties(ASTER PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${ASTER_BINARY_DIR}/python_packages/aster/_mlir_libs"
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ASTER PRIVATE "-Wl,-exclude-libs,ALL")
else()
  if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
    message(STATUS "ASTER on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
  endif()
endif()
