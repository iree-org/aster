add_mlir_library(AsterLSIRCodeGen
  CodeGenPatterns.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AsterCodeGen
  AsterInterfaces
  AsterTransforms
  AsterUtilsDialect
  LSIRDialect
  MLIRArithDialect
  MLIRIR
  MLIRTransformUtils
)
