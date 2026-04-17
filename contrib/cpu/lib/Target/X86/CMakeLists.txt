add_mlir_translation_library(ASTERX86Translate
  TranslateToAsm.cpp

  DEPENDS
  ASTERX86IncGen

  LINK_LIBS PUBLIC
  ASTERX86Dialect
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
)
