add_mlir_library(AsterUtilsTransforms
  AsterSelectiveInlining.cpp
  DestructureStructIterArgs.cpp
  FromAnyToPoison.cpp
  ResolveAnyIterArgs.cpp
  InlineExecuteRegion.cpp
  OptimizePtrAdd.cpp
  Transforms.cpp
  WrapCallsWithExecuteRegion.cpp

  DEPENDS
  AsterUtilsPassIncGen

  LINK_LIBS PUBLIC
  ASTERAnalysis
  AsterUtilsDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRUBDialect
)
