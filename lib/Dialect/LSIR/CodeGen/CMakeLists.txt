add_mlir_library(AsterLSIRCodeGen
  CodeGenPatterns.cpp

  LINK_LIBS PUBLIC
  AsterCodeGen
  AMDGCNDialect
  AsterUtilsDialect
  LSIRDialect
  AsterInterfaces
  AsterTransforms
  MLIRArithDialect
  MLIRIR
  MLIRTransformUtils
)
