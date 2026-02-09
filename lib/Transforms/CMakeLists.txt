add_mlir_library(AsterTransforms
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
  MLIRPass
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRUBDialect
  MLIRTransformUtils
)
