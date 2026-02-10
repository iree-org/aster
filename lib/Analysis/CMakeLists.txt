add_mlir_library(
  ASTERAnalysis
  MemoryDependenceAnalysis.cpp
  RegisterConstraints.cpp
  ThreadUniformAnalysis.cpp

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
