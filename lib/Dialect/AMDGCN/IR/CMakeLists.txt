add_mlir_dialect_library(AMDGCNDialect
  AMDGCN.cpp
  AMDGCNAttrs.cpp
  AMDGCNTypes.cpp
  AMDGCNVerifiers.cpp
  Utils.cpp

  DEPENDS
  AMDGCNAttrInterfacesIncGen
  AMDGCNInstOpInterfaceIncGen
  AMDGCNInstsIncGen
  AMDGCNOpsIncGen
  AMDGCNTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  ASTERIR
  AsterInterfaces
  LSIRDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRPtrDialect
)
