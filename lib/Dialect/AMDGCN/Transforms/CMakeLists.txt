add_mlir_library(
  AMDGCNTransforms
  AsterSelectiveInlining.cpp
  ConstexprExpansion.cpp
  InstructionScheduling.cpp
  InstructionSchedulingAutoschedulePass.cpp
  LowerToAMDGCN.cpp
  Mem2Reg.cpp
  RegisterAlloc.cpp
  RegisterDealloc.cpp
  ToAMDGCN.cpp
  ToAMDGCNPatterns.cpp

  DEPENDS
  AMDGCNPassesIncGen

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRUBDialect
  AsterTransforms
)
