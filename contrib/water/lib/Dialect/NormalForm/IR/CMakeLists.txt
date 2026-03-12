add_mlir_dialect_library(MLIRWaterNormalFormDialect
  NormalFormDialect.cpp
  NormalFormInterfaces.cpp
  NormalFormOps.cpp

  DEPENDS
  MLIRWaterNormalFormIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

# Install the Wave dialect library so Python can find it at runtime.
install(TARGETS MLIRWaterNormalFormDialect
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
