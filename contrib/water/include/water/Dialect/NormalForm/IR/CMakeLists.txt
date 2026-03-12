add_custom_target(MLIRWaterNormalFormIncGen)

set(LLVM_TARGET_DEFINITIONS NormalFormDialect.td)
mlir_tablegen(NormalFormDialect.h.inc -gen-dialect-decls -dialect=normalform)
mlir_tablegen(NormalFormDialect.cpp.inc -gen-dialect-defs -dialect=normalform)
add_public_tablegen_target(MLIRWaterNormalFormDialectIncGen)

set(LLVM_TARGET_DEFINITIONS NormalFormOps.td)
mlir_tablegen(NormalFormOps.h.inc -gen-op-decls)
mlir_tablegen(NormalFormOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRWaterNormalFormOpsIncGen)
add_dependencies(MLIRWaterNormalFormIncGen MLIRWaterNormalFormOpsIncGen)

set(LLVM_TARGET_DEFINITIONS NormalFormInterfaces.td)
mlir_tablegen(NormalFormAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(NormalFormAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRWaterNormalFormInterfacesIncGen)
add_dependencies(MLIRWaterNormalFormIncGen MLIRWaterNormalFormInterfacesIncGen)
add_mlir_doc(WaterNormalFormInterfaces WaterNormalFormAttrInterfaces Dialects/ -gen-attr-interface-docs)

add_dependencies(mlir-headers MLIRWaterNormalFormIncGen)

add_mlir_doc(WaterNormalForm WaterNormalForm Dialects/ -gen-dialect-doc -dialect normalform)
