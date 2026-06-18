add_mlir_library(AMDGCNScheduler
  GraphBuilderAttrs.cpp
  ILPSchedulerAttrs.cpp
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

target_link_libraries(AMDGCNScheduler PRIVATE ortools::ortools)
