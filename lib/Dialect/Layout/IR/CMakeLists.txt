add_mlir_dialect_library(LayoutDialect
  LayoutAttrs.cpp
  LayoutOps.cpp

  DEPENDS
  LayoutOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRSideEffectInterfaces
)
