add_mlir_dialect_library(AMDGCNDialect
  AMDGCN.cpp
  AMDGCNAttrs.cpp
  AMDGCNTypes.cpp
  Utils.cpp
  AMDGCNVerifiers.cpp

  ADDITIONAL_HEADER_DIRS
  ${AMDGCN_INCLUDE_DIR}/AMDGCN/Dialect/AMDGCN

  DEPENDS
  AMDGCNOpsIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRArithDialect
  MLIRPtrDialect
  AsterInterfaces
  )
