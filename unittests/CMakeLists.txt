# ASTER C++ unit tests (MLIR-style layout).
# Source lives here; executables go to build/unittests/ for lit discovery.
# Lit config is in test/Unit/.

if(NOT ASTER_ENABLE_UNIT_TESTS)
  return()
endif()

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Output directory for unit test executables (lit discovers them here)
set(ASTER_UNITTEST_DIR "${CMAKE_BINARY_DIR}/unittests")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${ASTER_UNITTEST_DIR}")

# For multi-config generators (Visual Studio), append build type to path
if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${ASTER_UNITTEST_DIR}/Debug")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${ASTER_UNITTEST_DIR}/Release")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${ASTER_UNITTEST_DIR}/RelWithDebInfo")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${ASTER_UNITTEST_DIR}/MinSizeRel")
endif()

function(add_aster_unittest test_name)
  add_executable(${test_name} ${ARGN})
  target_link_libraries(${test_name} PRIVATE
    GTest::gtest_main
    GTest::gtest
  )
  target_include_directories(${test_name} PRIVATE
    ${ASTER_SOURCE_DIR}/include
    ${ASTER_BINARY_DIR}/include
  )
  set_target_properties(${test_name} PROPERTIES FOLDER "Tests/Unit")
  add_dependencies(ASTERUnitTests ${test_name})
endfunction()

# Create the aggregate target that lit depends on
add_custom_target(ASTERUnitTests)

add_subdirectory(Support)
