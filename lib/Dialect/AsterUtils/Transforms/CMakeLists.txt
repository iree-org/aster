add_mlir_library(AsterUtilsTransforms
  AsterSelectiveInlining.cpp
  FromAnyToPoison.cpp
  InlineExecuteRegion.cpp
  Transforms.cpp
  WrapCallsWithExecuteRegion.cpp

  DEPENDS
  AsterUtilsPassIncGen

  LINK_LIBS PUBLIC
  AsterUtilsDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRUBDialect
)
