add_mlir_dialect_library(ASTERX86Dialect
  X86Dialect.cpp

  DEPENDS
  ASTERX86IncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
