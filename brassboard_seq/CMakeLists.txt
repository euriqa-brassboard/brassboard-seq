#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(brassboard_seq_utils SHARED
  src/utils.cpp)
target_include_directories(brassboard_seq_utils PRIVATE ${Python_INCLUDE_DIRS})
install(TARGETS brassboard_seq_utils DESTINATION "${CMAKE_INSTALL_LIBDIR}")

cython_link_libraries(brassboard_seq_utils)

python_install_module("${CMAKE_CURRENT_SOURCE_DIR}/.." brassboard_seq
  __init__.py
  rtval.pyx
  utils.pxd)
