add_mlir_library(
  AMDGCNTransforms

  AMDGCNNopInsertion.cpp
  AMDGCNOptimizeStraightLineWaits.cpp
  ConstexprExpansion.cpp
  ExpandMetadataOps.cpp
  InstructionScheduling.cpp
  InstructionSchedulingAutoschedulePass.cpp
  HoistOps.cpp
  Mem2Reg.cpp
  PreloadLibrary.cpp
  RegisterAlloc.cpp
  RegisterDealloc.cpp
  RemoveTestInst.cpp
  SetABI.cpp
  SelectRegClasses.cpp
  ToAMDGCN.cpp
  ToAMDGCNPatterns.cpp

  DEPENDS
  AMDGCNPassesIncGen

  LINK_LIBS PUBLIC
  AMDGCNDialect
  ASTERAnalysis
  AsterTransforms

  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRTransforms
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRUBDialect
)
