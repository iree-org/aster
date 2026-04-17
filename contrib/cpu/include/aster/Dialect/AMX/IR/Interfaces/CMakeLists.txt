set(LLVM_TARGET_DEFINITIONS AMXAsmOpInterface.td)
mlir_tablegen(AMXAsmOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(AMXAsmOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ASTERAMXAsmOpInterfaceIncGen)
