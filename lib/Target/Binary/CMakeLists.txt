if (ASTER_ENABLE_TARGET)
  add_compile_definitions(ASTER_ENABLE_TARGET=1)
  set(AMDGPU_LIBS
    Linker
    MC
    Target
    MCParser
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
  set(LLD_LIBS
    lldCommon
    lldELF
  )
else()
  add_compile_definitions(ASTER_ENABLE_TARGET=0)
endif()

add_mlir_library(
  AsterBinaryTarget
  CompileAsm.cpp

  LINK_COMPONENTS
  Support
  ${AMDGPU_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  PRIVATE
  ${LLD_LIBS}
)
