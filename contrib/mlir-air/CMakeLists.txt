add_mlir_library(MlirAirLib
  lib/AirToAMDGCN.cpp
  lib/ConvertToAMDGCNLibraryCalls.cpp
  lib/ConvertMemSpaceToAMDGCN.cpp
  lib/Init.cpp
  lib/Pipelines.cpp

  LINK_LIBS PUBLIC
  AIRConversionPasses
  AIRDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransformOps
  MLIRBufferizationTransforms
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTensorTransforms
  AIRTransformOps
  AIRTransformPasses
  ASTERInit
  AsterTransforms
  AsterCodeGen
  AMDGCNTransforms
  AsterUtilsTransforms
  LSIRDialect
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRPtrDialect
  MLIRSCFTransformOps
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
)

add_llvm_executable(mlir-air-opt
  tools/mlir-air-opt.cpp
)
llvm_update_compile_flags(mlir-air-opt)
target_link_libraries(mlir-air-opt PRIVATE
  MlirAirLib
  ASTERInit
  AMDGCNTransforms
  MLIROptLib
)
mlir_check_all_link_libraries(mlir-air-opt)
install(TARGETS mlir-air-opt)
