set(LLVM_TARGET_DEFINITIONS InstOpInterface.td)
mlir_tablegen(InstOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InstOpInterface.cpp.inc -gen-op-interface-defs)
set(LLVM_TARGET_DEFINITIONS RegisterType.td)
mlir_tablegen(RegisterType.h.inc -gen-type-interface-decls)
mlir_tablegen(RegisterType.cpp.inc -gen-type-interface-defs)

add_public_tablegen_target(AsterInterfacesIncGen)
