set(LIBS
  AMDGCNTransforms
  ASTERInit
  ASTERTestPass
  MLIRAsterTestDialect
  MLIROptLib
)

add_llvm_executable(aster-opt
  aster-opt.cpp
)

llvm_update_compile_flags(aster-opt)
target_link_libraries(aster-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(aster-opt)
if(DEFINED SKBUILD_PLATLIB_DIR)
  install(TARGETS aster-opt
    RUNTIME
      DESTINATION "${ASTER_TOOLS_INSTALL_PREFIX}"
      COMPONENT ASTERPythonModules
  )
else()
  install(TARGETS aster-opt)
endif()
