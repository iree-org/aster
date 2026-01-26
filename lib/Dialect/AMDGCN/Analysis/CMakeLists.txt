add_mlir_library(
  AMDGCNAnalysis
  WaitAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aster/Dialect/AMDGCN/Analysis

  LINK_LIBS PUBLIC
  AMDGCNDialect

  AMDGCNDialect
  MLIRAnalysis
  MLIRIR
  MLIRSupport
)
