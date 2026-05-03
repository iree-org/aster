set(LLVM_TARGET_DEFINITIONS KernelArgInterface.td)
mlir_tablegen(KernelArgInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(KernelArgInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(KernelArgInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS HazardAttrInterface.td)
mlir_tablegen(HazardAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(HazardAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(HazardAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS AMDGCNRegisterTypeInterface.td)
mlir_tablegen(AMDGCNRegisterTypeInterface.h.inc -gen-type-interface-decls)
mlir_tablegen(AMDGCNRegisterTypeInterface.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(AMDGCNRegisterTypeInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS AMDGCNInstOpInterface.td)
mlir_tablegen(AMDGCNInstOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(AMDGCNInstOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AMDGCNInstOpInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS MemoryOpInterfaces.td)
mlir_tablegen(MemoryOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(MemoryOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MemoryOpInterfacesIncGen)
