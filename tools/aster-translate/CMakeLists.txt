set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_tool(aster-translate
  aster-translate.cpp
  )
llvm_update_compile_flags(aster-translate)
mlir_target_link_libraries(aster-translate
  PRIVATE
  ASTERInit
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  AsterExportASM
  )

mlir_check_link_libraries(aster-translate)
