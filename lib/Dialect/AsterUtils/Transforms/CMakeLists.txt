add_mlir_library(AsterUtilsTransforms
  AsterSelectiveInlining.cpp
  DestructureStructIterArgs.cpp
  FromAnyToPoison.cpp
  InlineExecuteRegion.cpp
  OptimizePtrAdd.cpp
  ResolveAnyIterArgs.cpp
  Transforms.cpp
  WrapCallsWithExecuteRegion.cpp

  DEPENDS
  AsterUtilsPassIncGen

  LINK_LIBS PUBLIC
  ASTERAnalysis
  AsterTransforms
  AsterUtilsDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRUBDialect
)
