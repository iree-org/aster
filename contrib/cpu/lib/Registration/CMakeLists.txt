add_mlir_library(ASTERAMXRegistration
  Registration.cpp

  LINK_LIBS PUBLIC
  ASTERAMXDialect
  ASTERAMXTranslate
  MLIRIR
  MLIRTranslateLib
)
