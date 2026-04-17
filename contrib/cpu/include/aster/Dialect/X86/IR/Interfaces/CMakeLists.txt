set(LLVM_TARGET_DEFINITIONS X86IsaOpInterface.td)
mlir_tablegen(X86IsaOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(X86IsaOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ASTERX86IsaOpInterfaceIncGen)
