add_mlir_library(AsterBinaryTarget
  CompileAsm.cpp

  LINK_LIBS PUBLIC
  LLVMAMDGPUAsmParser
  LLVMAMDGPUCodeGen
  LLVMAMDGPUUtils
  LLVMAMDGPUDesc
  LLVMAMDGPUInfo
  LLVMLinker
  LLVMMC
  LLVMMCParser
  LLVMSupport
  LLVMTarget
  lldELF
  lldCommon
  MLIRIR
  MLIRSupport
)
