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
  AsterInterfaces
  AsterUtilsDialect
  LSIRDialect
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRTransformUtils
  MLIRUBDialect
  MLIRVectorDialect
  MLIRVectorTransforms
)
