add_mlir_library(
  AMDGCNAnalysis
  BufferAnalysis.cpp
  LDSInterferenceGraph.cpp
  RangeConstraintAnalysis.cpp
  ReachingDefinitions.cpp
  RegisterInterferenceGraph.cpp
  WaitAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aster/Dialect/AMDGCN/Analysis

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERAnalysis
  ASTERIR
  ASTERSupport
  MLIRAnalysis
  MLIRIR
  MLIRSupport
)
