# Add pytest configuration for CDNA tests

if(ASTER_ENABLE_PYTHON)
  # Find Python
  find_package(Python 3 COMPONENTS Interpreter REQUIRED)

  # Configure the pytest command using Python's -m pytest
  set(PYTEST_ARGS
    -m pytest
    ${CMAKE_CURRENT_SOURCE_DIR}
    -v
    --tb=short
  )

  # Set PYTHONPATH to include build and install python packages
  set(PYTHONPATH_ENV
    ${CMAKE_BINARY_DIR}/python_packages:${CMAKE_INSTALL_PREFIX}/python_packages
  )

  # Add pytest target
  add_custom_target(check-python-cdna
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${PYTHONPATH_ENV}"
            ${Python_EXECUTABLE} ${PYTEST_ARGS}
    DEPENDS AMDGCNPythonModules
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running CDNA Python tests"
  )

  set_target_properties(check-python-cdna PROPERTIES FOLDER "Tests")
endif()
