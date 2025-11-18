add_mlir_library(ASTERTestPass
  TestMemoryDependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERAnalysis
  MLIRAnalysis
  MLIRIR
  MLIRPass
)
