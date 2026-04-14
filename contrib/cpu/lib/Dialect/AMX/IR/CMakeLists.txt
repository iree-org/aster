add_mlir_dialect_library(ASTERAMXDialect
  AMXDialect.cpp
  AMXAsmOpInterface.cpp

  DEPENDS
  ASTERAMXIncGen
  ASTERAMXAsmOpInterfaceIncGen
  ASTERX86IncGen

  LINK_LIBS PUBLIC
  ASTERX86Dialect
  MLIRIR
  MLIRSupport
)
