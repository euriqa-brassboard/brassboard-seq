#

include_directories(${Python_NumPy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -fopenmp
  -Wno-missing-field-initializers)

add_library(brassboard_seq_utils SHARED
  src/action.cpp
  src/config.cpp
  src/event_time.cpp
  src/parampack.cpp
  src/rtprop.cpp
  src/rtval.cpp
  src/seq.cpp
  src/utils.cpp
  src/yaml.cpp)
target_include_directories(brassboard_seq_utils PRIVATE ${Python_INCLUDE_DIRS})
install(TARGETS brassboard_seq_utils DESTINATION "${CMAKE_INSTALL_LIBDIR}")

add_library(brassboard_seq__utils_mod MODULE
  _utils.cpp)
target_include_directories(brassboard_seq__utils_mod PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(brassboard_seq__utils_mod brassboard_seq_utils -fopenmp)
set_target_properties(brassboard_seq__utils_mod
    PROPERTIES POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_NAME "_utils"
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
install(TARGETS brassboard_seq__utils_mod
  DESTINATION "${_PYTHON_SITE_INSTALL_DIR}/brassboard_seq")

install(FILES
  src/action.h
  src/backend.h
  src/config.h
  src/artiq_backend.h
  src/event_time.h
  src/rtval.h
  src/scan.h
  src/seq.h
  src/rfsoc_backend.h
  src/utils.h
  DESTINATION "${_PYTHON_SITE_INSTALL_DIR}/brassboard_seq/src")

cython_link_libraries(brassboard_seq_utils -fopenmp)

python_install_module("${CMAKE_CURRENT_SOURCE_DIR}/.." brassboard_seq
  __init__.py
  action.pyx
  artiq_backend.pyx
  backend.pyx
  config.pyx
  event_time.pyx
  rfsoc_backend.pyx
  rtval.pyx
  scan.pyx
  seq.pyx
  utils.pxd
  yaml.pyx)
