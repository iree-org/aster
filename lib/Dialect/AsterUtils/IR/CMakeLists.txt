add_mlir_dialect_library(AsterUtilsDialect
  AsterUtilsOps.cpp
  AsterUtilsTypes.cpp
  AsterUtilsAttrs.cpp
  IntRangeImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${ASTER_INCLUDE_DIR}/Dialect/AsterUtils/IR

  DEPENDS
  AsterUtilsOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
