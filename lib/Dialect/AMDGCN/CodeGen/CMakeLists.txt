add_mlir_library(AsterAMDGCNCodeGen
  CodeGenPatterns.cpp

  LINK_LIBS PUBLIC
  AsterCodeGen
  AMDGCNDialect
  MLIRIR
  MLIRTransformUtils
)
