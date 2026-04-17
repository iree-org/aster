add_mlir_library(ASTERX86Registration
  Registration.cpp

  LINK_LIBS PUBLIC
  ASTERX86Dialect
  ASTERX86Translate
  MLIRIR
  MLIRTranslateLib
)
