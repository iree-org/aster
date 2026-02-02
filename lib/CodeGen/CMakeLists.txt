set(LLVM_OPTIONAL_SOURCES
  CodeGen.cpp
  CodeGenPass.cpp
)

add_mlir_library(AsterCodeGen
  CodeGen.cpp

  LINK_LIBS PUBLIC
  ASTERAnalysis
  LSIRDialect
  AsterInterfaces
  MLIRIR
  MLIRTransformUtils
)

add_mlir_library(AsterCodeGenTransforms
  CodeGenPass.cpp

  DEPENDS
  AsterCodeGenPassesIncGen

  LINK_LIBS PUBLIC
  AsterCodeGen
  AsterTransforms
  MLIRPass
  MLIRTransformUtils
  # TODO: Remove these when we use interfaces ala ConvertToLLVM.
  AsterLSIRCodeGen
  AsterAMDGCNCodeGen
)
