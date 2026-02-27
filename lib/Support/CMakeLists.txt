add_mlir_library(ASTERSupport
  Graph.cpp
  Lexer.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRSupport
)
