set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(ASTERTestPassesIncGen)

add_mlir_library(ASTERTestPass
  Passes.cpp
  TestAMDGCNInterferenceAnalysis.cpp
  TestCFGWalker.cpp
  TestDPSAnalysis.cpp
  TestLivenessAnalysis.cpp
  TestAMDGCNRangeAnalysis.cpp
  TestBufferAnalysis.cpp
  TestLDSInterferenceGraph.cpp
  TestLDSMultibufferPrep.cpp
  TestMemoryDependenceAnalysis.cpp
  TestWaitAnalysis.cpp

  DEPENDS
  ASTERTestPassesIncGen

  LINK_LIBS PUBLIC
  AMDGCNAnalysis
  AMDGCNDialect
  ASTERAnalysis
  AsterTransforms
  MLIRAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  ASTERIR
)

target_include_directories(ASTERTestPass PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
