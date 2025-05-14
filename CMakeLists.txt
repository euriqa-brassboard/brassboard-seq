cmake_minimum_required(VERSION 3.14)

set(MAJOR_VERSION 0)
set(MINOR_VERSION 0)
set(PATCH_VERSION 0)

set(CMAKE_CXX_STANDARD 20)

project(brassboard-seq VERSION "${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}")

## Setup
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(PythonMacros)

option(ENABLE_TESTING "Enable building and running testing" On)
add_definitions(-DPY_SSIZE_T_CLEAN -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
  -Wno-missing-field-initializers)

add_subdirectory(brassboard_seq)
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

add_custom_target(run_python
  COMMAND env PYTHONPATH="${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/tests" python
  DEPENDS all-python-target
  USES_TERMINAL)

add_custom_target(run_ipython
  COMMAND env PYTHONPATH="${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/tests" ipython
  DEPENDS all-python-target
  USES_TERMINAL)
