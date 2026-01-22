add_mlir_library(ASTERTestPass
  TestDPSAliasAnalysis.cpp
  TestLivenessAnalysis.cpp
  TestMemoryDependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERAnalysis
  MLIRAnalysis
  MLIRIR
  MLIRPass
)
