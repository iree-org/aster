add_mlir_library(ASTERTestPass
  TestDPSAliasAnalysis.cpp
  TestLivenessAnalysis.cpp
  TestMemoryDependenceAnalysis.cpp
  TestWaitAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNAnalysis
  AMDGCNDialect
  ASTERAnalysis
  MLIRAnalysis
  MLIRIR
  MLIRPass
)
