set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(ASTERTestPassesIncGen)

add_mlir_library(ASTERTestPass
  Passes.cpp
  TestBufferAnalysis.cpp
  TestDPSAliasAnalysis.cpp
  TestInterferenceAnalysis.cpp
  TestLDSInterferenceGraph.cpp
  TestLivenessAnalysis.cpp
  TestMemoryDependenceAnalysis.cpp
  TestRangeAnalysis.cpp
  TestValueProvenanceAnalysis.cpp
  TestWaitAnalysis.cpp

  DEPENDS
  ASTERTestPassesIncGen

  LINK_LIBS PUBLIC
  AMDGCNAnalysis
  AMDGCNDialect
  ASTERAnalysis
  MLIRAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)

target_include_directories(ASTERTestPass PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
