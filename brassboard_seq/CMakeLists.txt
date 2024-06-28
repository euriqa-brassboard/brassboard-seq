#

include_directories(${Python_NumPy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

python_install_module("${CMAKE_CURRENT_SOURCE_DIR}/.." brassboard_seq
  __init__.py
  action.pyx
  artiq_backend.pyx
  config.pyx
  event_time.pyx
  rtval.pyx
  scan.pyx
  seq.pyx
  utils.pxd)
