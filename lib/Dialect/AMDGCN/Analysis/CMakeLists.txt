add_mlir_library(AMDGCNAnalysis
  BufferAnalysis.cpp
  LDSInterferenceGraph.cpp
  RangeConstraintAnalysis.cpp
  ReachingDefinitions.cpp
  RegisterInterferenceGraph.cpp
  WaitAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERAnalysis
  ASTERIR
  ASTERSupport
  MLIRAnalysis
  MLIRIR
  MLIRSupport
)
