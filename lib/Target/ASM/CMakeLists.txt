set(LLVM_TARGET_DEFINITIONS AMDGCN.td)
amdgcn_tablegen(AMDGCNAsmPrinter.cpp.inc -gen-inst-printers)
add_public_tablegen_target(AMDGCNAsmIncGen)

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
)

add_mlir_library(
  AsterExportASM
  AsmPrinter.cpp
  TranslateModule.cpp

  DEPENDS
  AMDGCNAsmIncGen

  LINK_LIBS PUBLIC
  AMDGCNDialect
  MLIRArithDialect
)
