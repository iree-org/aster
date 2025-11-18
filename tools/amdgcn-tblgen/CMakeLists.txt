set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(amdgcn-tblgen Aster
  DESTINATION "${ASTER_TOOLS_INSTALL_DIR}"
  EXPORT Aster
  main.cpp
  Dump.cpp
  Instruction.cpp
  InstAsmFormat.cpp
  InstBindings.cpp
  InstCommon.cpp
  InstGen.cpp
  OpGen.cpp
)

target_link_libraries(amdgcn-tblgen
  PRIVATE
  MLIRTableGen
  ASTERSupport
)

mlir_check_all_link_libraries(amdgcn-tblgen)
