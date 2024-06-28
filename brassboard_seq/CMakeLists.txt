#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

python_install_module("${CMAKE_CURRENT_SOURCE_DIR}/.." brassboard_seq
  __init__.py
  utils.pxd)
