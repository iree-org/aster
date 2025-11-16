add_mlir_library(
  ASTERAnalysis
  InterferenceGraph.cpp
  LivenessAnalysis.cpp
  RangeAnalysis.cpp
  ThreadUniformAnalysis.cpp
  VariableAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aster/Analysis

  LINK_LIBS PUBLIC
  AMDGCNDialect

  ASTERSupport
  MLIRAnalysis
  MLIRIR
  MLIRSupport
)
