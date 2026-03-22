add_mlir_library(SchedTransforms
  ResourceExpansion.cpp
  GroupStagesToUnits.cpp
  InlineSchedUnits.cpp
  LoopPipeliner.cpp

  DEPENDS
  SchedPassIncGen

  LINK_LIBS PUBLIC
  AsterInterfaces
  AsterUtilsDialect
  SchedDialect
  MLIRIR
  MLIRLoopLikeInterface
  MLIRPass
  MLIRTransformUtils
)
