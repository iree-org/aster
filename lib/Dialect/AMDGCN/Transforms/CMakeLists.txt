add_mlir_library(
  AMDGCNTransforms

  AMDGCNNopInsertion.cpp
  Bufferization.cpp
  AMDGCNOptimizeStraightLineWaits.cpp
  ConvertLDSBuffers.cpp
  ConvertSCFControlFlow.cpp
  ConvertWaits.cpp
  ExpandMetadataOps.cpp
  InstructionSchedulingAutoschedulePass.cpp
  HoistOps.cpp
  LDSAlloc.cpp
  LegalizeCF.cpp
  LegalizeOperands.cpp
  Mem2Reg.cpp
  PreloadLibrary.cpp
  RegisterAlloc.cpp
  RegisterColoring.cpp
  RegisterDealloc.cpp
  RemoveTestInst.cpp
  SetABI.cpp
  ToAMDGCN.cpp
  ToAMDGCNPatterns.cpp

  DEPENDS
  AMDGCNPassesIncGen

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AMDGCNAnalysis
  ASTERAnalysis
  AsterTransforms

  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRTransforms
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRUBDialect
)
