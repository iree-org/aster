add_mlir_library(ASTERAnalysis
  DPSAnalysis.cpp
  LivenessAnalysis.cpp
  MemoryDependenceAnalysis.cpp
  RegisterConstraints.cpp
  ThreadUniformAnalysis.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERIR
  ASTERSupport
  AsterInterfaces
  AsterUtilsDialect
  LSIRDialect
  MLIRAnalysis
  MLIRGPUDialect
  MLIRIR
  MLIRSupport
)
