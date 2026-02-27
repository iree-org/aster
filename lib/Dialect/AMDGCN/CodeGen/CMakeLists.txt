add_mlir_library(AsterAMDGCNCodeGen
  CodeGenPatterns.cpp
  Legalizer.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AsterCodeGen
  AsterLegalizer
  AsterUtilsDialect
  LSIRDialect
  MLIRIR
  MLIRTransformUtils
)
