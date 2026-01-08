add_mlir_library(AsterTransforms
  ConstexprExpansion.cpp
  OpScheduling.cpp
  OptimizeArith.cpp
  ReplaceConstantGPUDims.cpp
  ToIntArith.cpp
  Utils.cpp

  DEPENDS
  AsterPassesIncGen

  LINK_LIBS PUBLIC
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
