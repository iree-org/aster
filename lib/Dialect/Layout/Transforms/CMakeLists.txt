add_mlir_library(LayoutTransforms
  LowerLayoutToAffine.cpp

  DEPENDS
  LayoutPassIncGen

  LINK_LIBS PUBLIC
  LayoutDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
