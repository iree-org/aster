add_mlir_library(
  ASTERAnalysis
  DPSAnalysis.cpp
  LivenessAnalysis.cpp
  MemoryDependenceAnalysis.cpp
  RegisterConstraints.cpp
  ThreadUniformAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aster/Analysis

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AsterInterfaces

  ASTERSupport
  AsterInterfaces
  ASTERIR
  MLIRAnalysis
  MLIRGPUDialect
  MLIRIR
  MLIRSupport
)
