add_mlir_library(AsterUtilsTransforms
  WrapCallsWithExecuteRegion.cpp
  InlineExecuteRegion.cpp

  DEPENDS
  AsterUtilsPassIncGen

  LINK_LIBS PUBLIC
  AsterUtilsDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
