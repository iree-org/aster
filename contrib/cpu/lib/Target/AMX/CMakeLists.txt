add_mlir_library(ASTERAMXTranslate
  TranslateToAsm.cpp

  LINK_LIBS PUBLIC
  ASTERAMXDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
