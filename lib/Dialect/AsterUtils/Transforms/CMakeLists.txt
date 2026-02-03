add_mlir_library(AsterUtilsTransforms
  AsterSelectiveInlining.cpp
  FromAnyToPoison.cpp
  InlineExecuteRegion.cpp
  OptimizePtrAdd.cpp
  Transforms.cpp
  WrapCallsWithExecuteRegion.cpp

  DEPENDS
  AsterUtilsPassIncGen

  LINK_LIBS PUBLIC
  ASTERAnalysis
  AsterUtilsDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRPtrDialect
  MLIRTransformUtils
  MLIRUBDialect
)
