add_mlir_dialect_library(MLIRPatternDialect
  PatternDialect.cpp
  PatternOps.cpp

  DEPENDS
  MLIRPatternIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIREmitCDialect
  MLIRTargetCpp
)
