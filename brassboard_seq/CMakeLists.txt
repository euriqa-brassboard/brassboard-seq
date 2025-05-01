#

include_directories(${Python_NumPy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -fopenmp
  -Wno-missing-field-initializers)

add_library(brassboard_seq_utils SHARED
  src/action.cpp
  src/config.cpp
  src/event_time.cpp
  src/init.cpp
  src/parampack.cpp
  src/rfsoc.cpp
  src/rtprop.cpp
  src/rtval.cpp
  src/scan.cpp
  src/seq.cpp
  src/utils.cpp
  src/yaml.cpp)
target_include_directories(brassboard_seq_utils PRIVATE ${Python_INCLUDE_DIRS})
install(TARGETS brassboard_seq_utils DESTINATION "${CMAKE_INSTALL_LIBDIR}")

function(bb_cxx_mod name)
  add_library("brassboard_seq_${name}_mod" MODULE "${name}.cpp")
  target_include_directories("brassboard_seq_${name}_mod" PRIVATE ${Python_INCLUDE_DIRS})
  target_link_libraries("brassboard_seq_${name}_mod" brassboard_seq_utils -fopenmp)
  set_target_properties("brassboard_seq_${name}_mod"
    PROPERTIES POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_NAME "${name}"
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
  install(TARGETS "brassboard_seq_${name}_mod"
    DESTINATION "${_PYTHON_SITE_INSTALL_DIR}/brassboard_seq")
endfunction()
bb_cxx_mod(_utils)
bb_cxx_mod(action)
bb_cxx_mod(config)
bb_cxx_mod(scan)
bb_cxx_mod(yaml)

install(FILES
  src/action.h
  src/backend.h
  src/config.h
  src/artiq_backend.h
  src/event_time.h
  src/rtval.h
  src/scan.h
  src/seq.h
  src/rfsoc.h
  src/rfsoc_backend.h
  src/utils.h
  DESTINATION "${_PYTHON_SITE_INSTALL_DIR}/brassboard_seq/src")

cython_link_libraries(brassboard_seq_utils -fopenmp)

python_install_module("${CMAKE_CURRENT_SOURCE_DIR}/.." brassboard_seq
  __init__.py
  action.pxd
  artiq_backend.pyx
  backend.pyx
  config.pxd
  event_time.pyx
  rfsoc_backend.pyx
  rtval.pyx
  seq.pyx
  utils.pxd)
