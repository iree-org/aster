add_subdirectory(Analysis)
add_subdirectory(API)
add_subdirectory(IR)
add_subdirectory(CodeGen)
add_subdirectory(Dialect)
add_subdirectory(Interfaces)
add_subdirectory(Support)
add_subdirectory(Target)
add_subdirectory(Transforms)

add_mlir_library(
  ASTERInit
  Init.cpp
  ENABLE_AGGREGATION

  DEPENDS
  AMDGCNOpsIncGen
  AMDGCNPassesIncGen

  LINK_LIBS PRIVATE
  AMDGCNDialect
  AMDGCNTransforms
  AsterCodeGenTransforms
  LSIRDialect
  AsterUtilsDialect
  AsterUtilsTransforms
  ASTERIR

  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRArithDialect
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRCastInterfaces
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRIndexDialect
  MLIRGPUDialect
  AsterInterfaces
  AsterUpstreamModelsImpl
)
