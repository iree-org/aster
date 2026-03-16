add_mlir_library(AsterUtilsTransforms
  RaiseToAffine.cpp
  DecomposeByLoopInvariant.cpp
  RemoveAssumeOps.cpp
  AsterSelectiveInlining.cpp
  DestructureStructIterArgs.cpp
  ExpandAffineApply.cpp
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
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRUBDialect
)
