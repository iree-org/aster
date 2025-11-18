add_mlir_dialect_library(LSIRDialect
  LSIROps.cpp
  LSIRTypes.cpp
  LSIRAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${ASTER_INCLUDE_DIR}/LSIR/Dialect/LSIR/IR

  DEPENDS
  LSIROpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
