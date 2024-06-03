#

include_directories(${Python_NumPy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

add_library(brassboard_seq_utils SHARED
  src/utils.cpp)
target_include_directories(brassboard_seq_utils PRIVATE ${Python_INCLUDE_DIRS})
install(TARGETS brassboard_seq_utils DESTINATION "${CMAKE_INSTALL_LIBDIR}")

cython_link_libraries(brassboard_seq_utils)

python_install_module("${CMAKE_CURRENT_SOURCE_DIR}/.." brassboard_seq
  __init__.py
  action.pyx
  config.pyx
  event_time.pyx
  rtval.pyx
  utils.pxd)
