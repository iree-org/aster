set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  MLIRLspServerLib

  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  ASTERInit
  )

add_mlir_tool(aster-lsp-server
  aster-lsp-server.cpp
  )
mlir_target_link_libraries(aster-lsp-server PRIVATE ${LIBS})
target_link_libraries(aster-lsp-server PRIVATE ${test_libs})
llvm_update_compile_flags(aster-lsp-server)

mlir_check_all_link_libraries(aster-lsp-server)
