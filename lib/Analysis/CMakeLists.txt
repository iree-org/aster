add_mlir_library(
  ASTERAnalysis
  InterferenceGraph.cpp
  LivenessAnalysis.cpp
  MemoryDependenceAnalysis.cpp
  RangeAnalysis.cpp
  RegisterConstraints.cpp
  ThreadUniformAnalysis.cpp
  DPSAliasAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aster/Analysis

  LINK_LIBS PUBLIC
  AMDGCNDialect

  ASTERSupport
  MLIRAnalysis
  MLIRGPUDialect
  MLIRIR
  MLIRSupport
)
