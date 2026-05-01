add_custom_target(MLIRPatternIncGen)

set(LLVM_TARGET_DEFINITIONS PatternDialect.td)
mlir_tablegen(PatternDialect.h.inc -gen-dialect-decls -dialect=pattern)
mlir_tablegen(PatternDialect.cpp.inc -gen-dialect-defs -dialect=pattern)
add_public_tablegen_target(MLIRPatternDialectIncGen)

set(LLVM_TARGET_DEFINITIONS PatternOps.td)
mlir_tablegen(PatternOps.h.inc -gen-op-decls)
mlir_tablegen(PatternOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRPatternOpsIncGen)
add_dependencies(MLIRPatternIncGen MLIRPatternOpsIncGen)

add_dependencies(mlir-headers MLIRPatternIncGen)

add_mlir_doc(PatternOps Pattern Dialects/ -gen-dialect-doc -dialect pattern)
