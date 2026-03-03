add_mlir_dialect_library(MLIRNormalFormTransforms
  LowerNormalFormModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aster/

  DEPENDS
  MLIRAsterNormalFormPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRNormalFormDialect
  MLIRParser
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
