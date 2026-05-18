set(LIBS
  AMDGCNTransforms
  ASTERInit
  ASTERTestPass
  MLIROptLib
)

add_llvm_executable(aster-opt
  aster-opt.cpp
)

llvm_update_compile_flags(aster-opt)
target_link_libraries(aster-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(aster-opt)
install(TARGETS aster-opt)
