add_mlir_library(ASTERTestPass
  TestBufferAnalysis.cpp
  TestDPSAliasAnalysis.cpp
  TestLivenessAnalysis.cpp
  TestMemoryDependenceAnalysis.cpp
  TestWaitAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNAnalysis
  AMDGCNDialect
  ASTERAnalysis
  MLIRAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
