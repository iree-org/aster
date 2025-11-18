
add_mlir_library(AsterTransforms
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRTransformUtils
)
