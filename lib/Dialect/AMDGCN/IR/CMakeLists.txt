add_mlir_dialect_library(AMDGCNDialect
  AMDGCN.cpp
  AMDGCNAttrs.cpp
  Sched.cpp
  SchedAttrs.cpp
  AMDGCNTypes.cpp
  AMDGCNVerifiers.cpp
  InstAttr.cpp
  Hazards.cpp
  Utils.cpp

  DEPENDS
  KernelArgInterfaceIncGen
  HazardAttrInterfaceIncGen
  AMDGCNInstOpInterfaceIncGen
  SchedInterfacesIncGen
  AMDGCNInstsIncGen
  AMDGCNOpsIncGen
  AMDGCNRegisterTypeInterfaceIncGen
  HazardAttrInterfaceIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
  ASTERIR
  AsterInterfaces
  LSIRDialect
  MLIRNormalFormDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRPtrDialect
)
