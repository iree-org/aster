add_mlir_library(
  AMDGCNAnalysis
  BufferAnalysis.cpp
  LDSInterferenceGraph.cpp
  RangeConstraintAnalysis.cpp
  RegisterInterferenceGraph.cpp
  RegisterLiveness.cpp
  Utils.cpp
  WaitAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aster/Dialect/AMDGCN/Analysis

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERIR
  ASTERSupport

  AMDGCNDialect
  MLIRAnalysis
  MLIRIR
  MLIRSupport
)
