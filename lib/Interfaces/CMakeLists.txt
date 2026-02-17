set(LLVM_OPTIONAL_SOURCES
  UpstreamModelsImpl.cpp
  AllocaOpInterface.cpp
  DependentOpInterface.cpp
  GPUFuncInterface.cpp
  InstOpInterface.cpp
  MemorySpaceConstraints.cpp
  ModuleOpInterface.cpp
  OperandBundleOpInterface.cpp
  RegisterType.cpp
  ResourceInterfaces.cpp
  VerifierAttr.cpp
)

add_mlir_library(AsterInterfaces
  AllocaOpInterface.cpp
  GPUFuncInterface.cpp
  DependentOpInterface.cpp
  InstOpInterface.cpp
  MemorySpaceConstraints.cpp
  ModuleOpInterface.cpp
  OperandBundleOpInterface.cpp
  RegisterType.cpp
  ResourceInterfaces.cpp
  VerifierAttr.cpp

  ADDITIONAL_HEADER_DIRS
  ${ASTER_INCLUDE_DIR}/aster/Interfaces

  DEPENDS
  AsterInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

add_mlir_library(AsterUpstreamModelsImpl
  UpstreamModelsImpl.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  AsterInterfaces
)
