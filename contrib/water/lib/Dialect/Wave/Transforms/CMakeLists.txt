add_mlir_dialect_library(MLIRWaveTransforms
  Asterize.cpp
  InferTypes.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/water/

  DEPENDS
  MLIRWaterWavePassesIncGen

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AsterExportASM
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransformDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRWaveDialect
)
