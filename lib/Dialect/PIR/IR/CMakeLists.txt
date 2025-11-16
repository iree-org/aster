add_mlir_dialect_library(PIRDialect
  PIROps.cpp
  PIRTypes.cpp
  PIRAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${ASTER_INCLUDE_DIR}/PIR/Dialect/PIR/IR

  DEPENDS
  PIROpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
