add_mlir_library(ASTERTestPass
  TestConstantMemoryOffsetAnalysis.cpp
  TestMemoryDependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERAnalysis
  MLIRAnalysis
  MLIRIR
  MLIRPass
)
