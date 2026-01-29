add_mlir_library(
  AMDGCNTransforms

  AMDGCNNopInsertion.cpp
  AMDGCNOptimizeStraightLineWaits.cpp
  ConvertLDSBuffers.cpp
  ConvertSCFControlFlow.cpp
  ConvertWaits.cpp
  ExpandMetadataOps.cpp
  InstructionSchedulingAutoschedulePass.cpp
  HoistOps.cpp
  LDSAlloc.cpp
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
  AMDGCNAnalysis
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
