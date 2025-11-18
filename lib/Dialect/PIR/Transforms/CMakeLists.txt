add_mlir_library(PIRTransforms
  ToPIR.cpp
  ToPIRPatterns.cpp

  DEPENDS
  PIRPassIncGen

  LINK_LIBS PUBLIC
  PIRDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  ASTERAnalysis
)
