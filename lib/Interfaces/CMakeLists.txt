
add_mlir_library(AsterInterfaces
  InstOpInterface.cpp
  RegisterType.cpp

  ADDITIONAL_HEADER_DIRS
  ${ASTER_INCLUDE_DIR}/aster/Interfaces

  DEPENDS
  AsterInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
