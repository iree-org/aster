set(LLVM_OPTIONAL_SOURCES
  AllocaOpInterface.cpp
  DependentOpInterface.cpp
  GPUFuncInterface.cpp
  InstOpInterface.cpp
  LivenessOpInterface.cpp
  MemorySpaceConstraints.cpp
  ModuleOpInterface.cpp
  OperandBundleOpInterface.cpp
  RegisterType.cpp
  ResourceInterfaces.cpp
  SchedInterfaces.cpp
  TargetAttr.cpp
  UpstreamModelsImpl.cpp
  VerifierAttr.cpp
)

add_mlir_library(AsterInterfaces
  AllocaOpInterface.cpp
  DependentOpInterface.cpp
  GPUFuncInterface.cpp
  InstOpInterface.cpp
  LivenessOpInterface.cpp
  MemorySpaceConstraints.cpp
  ModuleOpInterface.cpp
  OperandBundleOpInterface.cpp
  RegisterType.cpp
  ResourceInterfaces.cpp
  SchedInterfaces.cpp
  TargetAttr.cpp
  VerifierAttr.cpp

  DEPENDS
  AllocaOpInterfaceIncGen
  AsterInterfacesIncGen
  GPUFuncInterfaceIncGen
  InstOpInterfaceIncGen
  LivenessOpInterfaceIncGen
  MemorySpaceConstraintsIncGen
  ModuleOpInterfaceIncGen
  OperandBundleOpInterfaceIncGen
  RegisterSemanticsIncGen
  RegisterTypeIncGen
  ResourceInterfacesIncGen
  SchedInterfacesIncGen
  TargetAttrIncGen
  VerifierAttrIncGen

  LINK_LIBS PUBLIC
  ASTERSupport
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
)

add_mlir_library(AsterUpstreamModelsImpl
  UpstreamModelsImpl.cpp

  LINK_LIBS PUBLIC
  AsterInterfaces
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRPtrDialect
)
