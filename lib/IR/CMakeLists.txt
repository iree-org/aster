add_mlir_library(ASTERIR
  PrintingUtils.cpp
  SSAMap.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
