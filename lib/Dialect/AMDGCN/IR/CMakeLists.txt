add_mlir_dialect_library(AMDGCNDialect
  AMDGCN.cpp
  AMDGCNAttrs.cpp
  AMDGCNTypes.cpp
  AMDGCNVerifiers.cpp
  Hazards.cpp
  Utils.cpp

  DEPENDS
  KernelArgInterfaceIncGen
  HazardAttrInterfaceIncGen
  AMDGCNInstOpInterfaceIncGen
  InstOpInterfacesIncGen
  SchedInterfacesIncGen
  AMDGCNOpsIncGen
  AMDGCNRegisterTypeInterfaceIncGen
  HazardAttrInterfaceIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
  ASTERIR
  AsterInterfaces
  LSIRDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRPtrDialect
  MLIRTransformDialectInterfaces
  MLIRTransformDialectUtils
)
