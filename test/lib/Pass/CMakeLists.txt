add_mlir_library(ASTERTestPass
  TestLivenessAnalysis.cpp
  TestMemoryDependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERAnalysis
  MLIRAnalysis
  MLIRIR
  MLIRPass
)
