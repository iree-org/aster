set(LLVM_TARGET_DEFINITIONS AsterTestDialect.td)
mlir_tablegen(AsterTestDialect.h.inc -gen-dialect-decls -dialect=aster_test)
mlir_tablegen(AsterTestDialect.cpp.inc -gen-dialect-defs -dialect=aster_test)
mlir_tablegen(AsterTestDialectOps.h.inc -gen-op-decls)
mlir_tablegen(AsterTestDialectOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAsterTestDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TestNormalFormAttr.td)
mlir_tablegen(TestNormalFormAttr.h.inc -gen-attrdef-decls -attrdefs-dialect=aster_test)
mlir_tablegen(TestNormalFormAttr.cpp.inc -gen-attrdef-defs -attrdefs-dialect=aster_test)
add_public_tablegen_target(MLIRTestNormalFormAttrIncGen)

add_mlir_dialect_library(MLIRAsterTestDialect
  AsterTestDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestNormalFormAttrIncGen

  LINK_LIBS
  MLIRArithDialect
  MLIRIR
  MLIRNormalFormDialect
)

target_include_directories(MLIRAsterTestDialect
  PRIVATE
  ${PROJECT_BINARY_DIR}/test/lib/Dialect
)
