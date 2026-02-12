add_mlir_library(ASTERIR
  InstImpl.cpp
  PrintingUtils.cpp
  SSAMap.cpp

  LINK_LIBS PUBLIC
  AsterInterfaces
  MLIRIR
  MLIRSupport
)
