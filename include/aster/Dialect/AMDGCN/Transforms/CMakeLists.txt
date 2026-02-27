set(LLVM_TARGET_DEFINITIONS AMDGCNPasses.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name=AMDGCN)
add_public_tablegen_target(AMDGCNPassesIncGen)
