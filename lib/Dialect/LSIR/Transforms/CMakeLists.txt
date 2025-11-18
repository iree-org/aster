add_mlir_library(LSIRTransforms
  ToLSIR.cpp
  ToLSIRPatterns.cpp

  DEPENDS
  LSIRPassIncGen

  LINK_LIBS PUBLIC
  LSIRDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  ASTERAnalysis
)
