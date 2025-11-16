add_mlir_library(
  AsterAPI
  API.cpp
  ENABLE_AGGREGATION

  DEPENDS
  AMDGCNOpsIncGen

  LINK_LIBS PRIVATE
  AMDGCNDialect
  AsterExportASM
  AsterBinaryTarget
)
