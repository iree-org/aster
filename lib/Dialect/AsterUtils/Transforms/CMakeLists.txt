add_mlir_library(AsterUtilsTransforms
  AsterSelectiveInlining.cpp
  DestructureStructIterArgs.cpp
  FromAnyToPoison.cpp
  ResolveAnyIterArgs.cpp
  InlineExecuteRegion.cpp
  Transforms.cpp
  WrapCallsWithExecuteRegion.cpp

  DEPENDS
  AsterUtilsPassIncGen

  LINK_LIBS PUBLIC
  AsterUtilsDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRUBDialect
)
