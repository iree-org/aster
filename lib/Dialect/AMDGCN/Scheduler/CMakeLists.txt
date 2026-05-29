add_mlir_library(AMDGCNScheduler
  GraphBuilderAttrs.cpp
  SchedulerAttrs.cpp
  SchedulerCostModel.cpp

  DEPENDS
  SchedInterfacesIncGen

  LINK_LIBS PUBLIC
  AMDGCNDialect
  AMDGCNAnalysis
  AsterInterfaces
  LLVMSupport
  MLIRAnalysis
  MLIRIR
  MLIRSupport
)
