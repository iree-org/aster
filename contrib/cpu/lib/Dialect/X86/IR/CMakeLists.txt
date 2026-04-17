add_mlir_dialect_library(ASTERX86Dialect
  X86Dialect.cpp
  X86Ops.cpp

  DEPENDS
  ASTERX86IncGen
  ASTERX86IsaOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
)
