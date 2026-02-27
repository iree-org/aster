add_mlir_dialect_library(LSIRDialect
  LSIRAttrs.cpp
  LSIROps.cpp
  LSIRTypes.cpp

  DEPENDS
  LSIREnumsIncGen
  LSIROpsIncGen

  LINK_LIBS PUBLIC
  ASTERIR
  AsterInterfaces
  MLIRArithDialect
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRPtrDialect
  MLIRSideEffectInterfaces
)
