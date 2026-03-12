add_mlir_dialect_library(MLIRWaterNormalFormTransforms
  LowerNormalFormModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water/

  DEPENDS
  MLIRWaterNormalFormPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRWaterNormalFormDialect
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
