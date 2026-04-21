add_mlir_dialect_library(MLIRWaveTransforms
  Asterize.cpp
  DetectNormalForms.cpp
  ExpandVariadicReductions.cpp
  InferTypes.cpp
  LoweringPatterns.cpp
  LowerReadWriteOps.cpp
  LowerTransformPayload.cpp
  LowerWaveToMLIR.cpp
  PropagateDefaultsFromConstraints.cpp
  ResolveDistributedAllocations.cpp
  TypeConverter.cpp
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
