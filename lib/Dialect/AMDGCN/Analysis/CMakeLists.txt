add_mlir_library(
  AMDGCNAnalysis
  BufferAnalysis.cpp
  LDSInterferenceGraph.cpp
  WaitAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aster/Dialect/AMDGCN/Analysis

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERSupport

  AMDGCNDialect
  MLIRAnalysis
  MLIRIR
  MLIRSupport
)
