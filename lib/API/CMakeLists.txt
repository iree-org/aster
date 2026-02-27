add_mlir_library(AsterAPI
  API.cpp

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AsterBinaryTarget
  AsterExportASM
)
