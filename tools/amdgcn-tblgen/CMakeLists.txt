set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(amdgcn-tblgen Aster
  DESTINATION "${ASTER_TOOLS_INSTALL_DIR}"
  EXPORT Aster
  Dump.cpp
  InstAsmFormat.cpp
  InstBindings.cpp
  InstCommon.cpp
  InstCommon.h
  InstGen.cpp
  Instruction.cpp
  Instruction.h
  main.cpp
  OpGen.cpp
)

target_link_libraries(amdgcn-tblgen
  PRIVATE
  ASTERSupport
  MLIRTableGen
)

mlir_check_all_link_libraries(amdgcn-tblgen)
