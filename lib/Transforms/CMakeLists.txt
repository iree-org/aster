add_mlir_library(AsterTransforms
  CanonicalizePtrs.cpp
  ConstexprExpansion.cpp
  LDSMultibufferPrep.cpp
  OpScheduling.cpp
  OptimizeArith.cpp
  ReplaceConstantGPUDims.cpp
  SCFPipelineAsterSched.cpp
  ToIntArith.cpp
  Utils.cpp

  DEPENDS
  AsterPassesIncGen

  LINK_LIBS PUBLIC
  AMDGCNDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRAffineToStandard
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRUBDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorTransforms
)
