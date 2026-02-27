set(LIBS
  ASTERInit
  AsterExportASM
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTranslateLib
)

add_llvm_executable(aster-translate
  aster-translate.cpp
)

llvm_update_compile_flags(aster-translate)
target_link_libraries(aster-translate PRIVATE ${LIBS})
mlir_check_all_link_libraries(aster-translate)
install(TARGETS aster-translate)
