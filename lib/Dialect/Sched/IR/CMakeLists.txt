add_mlir_dialect_library(SchedDialect
  SchedOps.cpp

  DEPENDS
  SchedOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRTransformUtils
)
