add_mlir_dialect_library(AsterUtilsDialect
  AsterUtilsAttrs.cpp
  AsterUtilsOps.cpp
  AsterUtilsTypes.cpp
  IntRangeImpl.cpp

  DEPENDS
  AsterUtilsEnumsIncGen
  AsterUtilsOpsIncGen

  LINK_LIBS PUBLIC
  AsterInterfaces
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRPtrDialect
  MLIRSideEffectInterfaces
  MLIRTransformUtils
)
