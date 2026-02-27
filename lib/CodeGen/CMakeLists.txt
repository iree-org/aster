set(LLVM_OPTIONAL_SOURCES
  CodeGen.cpp
  CodeGenPass.cpp
  Legalizer.cpp
  MemOpsLegalization.cpp
  TypeConverter.cpp
  VectorLegalization.cpp
)

add_mlir_library(AsterCodeGen
  CodeGen.cpp

  LINK_LIBS PUBLIC
  ASTERAnalysis
  AsterInterfaces
  LSIRDialect
  MLIRIR
  MLIRTransformUtils
)

add_mlir_library(AsterLegalizer
  MemOpsLegalization.cpp
  TypeConverter.cpp
  VectorLegalization.cpp

  LINK_LIBS PUBLIC
  AsterTransforms
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRArithDialect
  MLIRArithUtils
  MLIRMemRefDialect
  MLIRPtrDialect
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorTransforms
)

add_mlir_library(AsterCodeGenTransforms
  CodeGenPass.cpp
  Legalizer.cpp

  DEPENDS
  AsterCodeGenPassesIncGen

  LINK_LIBS PUBLIC
  AsterCodeGen
  AsterLegalizer
  AsterTransforms
  MLIRPass
  AsterAMDGCNCodeGen
  AsterLSIRCodeGen
)
