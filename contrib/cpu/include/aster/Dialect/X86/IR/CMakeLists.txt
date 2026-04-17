add_subdirectory(Interfaces)

set(LLVM_TARGET_DEFINITIONS X86Ops.td)
mlir_tablegen(X86Dialect.h.inc -gen-dialect-decls -dialect=x86)
mlir_tablegen(X86Dialect.cpp.inc -gen-dialect-defs -dialect=x86)
mlir_tablegen(X86Enums.h.inc -gen-enum-decls)
mlir_tablegen(X86Enums.cpp.inc -gen-enum-defs)
mlir_tablegen(X86Types.h.inc -gen-typedef-decls -typedefs-dialect=x86)
mlir_tablegen(X86Types.cpp.inc -gen-typedef-defs -typedefs-dialect=x86)
mlir_tablegen(X86Attrs.h.inc -gen-attrdef-decls -attrdefs-dialect=x86)
mlir_tablegen(X86Attrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=x86)
mlir_tablegen(X86Ops.h.inc -gen-op-decls)
mlir_tablegen(X86Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ASTERX86IncGen)
add_dependencies(ASTERX86IncGen
  ASTERX86IsaOpInterfaceIncGen
  ASTERAMXAsmOpInterfaceIncGen)
