add_mlir_library(AsterUtilsTransforms
  RaiseToAffine.cpp
  DecomposeByLoopInvariant.cpp
  DecomposeByCSE.cpp
  RemoveAssumeOps.cpp
  AsterSelectiveInlining.cpp
  DestructureStructIterArgs.cpp
  ExpandAffineApply.cpp
  FromAnyToPoison.cpp
  OptimizePtrAdd.cpp
  ResolveAnyIterArgs.cpp
  Transforms.cpp

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
