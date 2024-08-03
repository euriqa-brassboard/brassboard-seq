#

include_directories(${Python_NumPy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

add_library(brassboard_seq_utils SHARED
  src/utils.cpp)
target_include_directories(brassboard_seq_utils PRIVATE ${Python_INCLUDE_DIRS})
install(TARGETS brassboard_seq_utils DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES
  src/action.h
  src/artiq_backend.h
  src/event_time.h
  src/rfsoc_backend.h
  src/rtval.h
  src/utils.h
  DESTINATION "${_PYTHON_SITE_INSTALL_DIR}/brassboard_seq/src")

cython_link_libraries(brassboard_seq_utils)

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
  utils.pxd)
