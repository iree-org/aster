add_mlir_library(AsterAMDGCNCodeGen
  CodeGenPatterns.cpp
  Legalizer.cpp

  LINK_LIBS PUBLIC
  AsterCodeGen
  AsterLegalizer
  AMDGCNDialect
  MLIRIR
  MLIRTransformUtils
)
