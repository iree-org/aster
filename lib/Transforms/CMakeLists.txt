add_mlir_library(AsterTransforms
  OptimizeArith.cpp
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
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRUBDialect
  MLIRTransformUtils
)
