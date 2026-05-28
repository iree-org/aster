# OR-Tools is built in-tree from the submodule. Force the options that keep
# the build focused: only the CP-SAT solver library, no samples, examples, or
# tests, and all transitive dependencies fetched automatically.
# Must be added after configure_llvm() to avoid OR-Tools FetchContent deps
# (Abseil, Protobuf, etc.) corrupting CMAKE_MODULE_PATH before LLVM is set up.
set(BUILD_DEPS ON CACHE BOOL "" FORCE)
set(BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Clp and Cbc (fetched transitively by OR-Tools) use __DATE__/__TIME__, which
# trips -Werror=date-time that LLVM injects into CMAKE_CXX_FLAGS. Strip it for
# the duration of the add_subdirectory call and restore it afterward so the
# flag still applies to all Aster code.
set(_saved_cxx_flags "${CMAKE_CXX_FLAGS}")
string(REPLACE "-Werror=date-time" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_subdirectory(or-tools)
set(CMAKE_CXX_FLAGS "${_saved_cxx_flags}")
