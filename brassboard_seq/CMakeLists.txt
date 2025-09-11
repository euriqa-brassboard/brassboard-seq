#

include_directories(${Python_NumPy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

if(HAS_OPENMP)
  add_definitions(-fopenmp -DHAS_OPENMP)
endif()

add_library(brassboard_seq_utils SHARED
  src/action.cpp
  src/artiq_backend.cpp
  src/artiq_utils.cpp
  src/backend.cpp
  src/config.cpp
  src/event_time.cpp
  src/init.cpp
  src/parampack.cpp
  src/rfsoc.cpp
  src/rfsoc_backend.cpp
  src/rfsoc_gen.cpp
  src/rtprop.cpp
  src/rtval.cpp
  src/scan.cpp
  src/seq.cpp
  src/utils.cpp
  src/yaml.cpp)
target_include_directories(brassboard_seq_utils PRIVATE ${Python_INCLUDE_DIRS})
if(HAS_OPENMP)
  target_link_libraries(brassboard_seq_utils -fopenmp)
endif()
install(TARGETS brassboard_seq_utils DESTINATION "${CMAKE_INSTALL_LIBDIR}")

function(bb_cxx_mod name)
  add_library("brassboard_seq_${name}_mod" MODULE "${name}.cpp")
  target_include_directories("brassboard_seq_${name}_mod" PRIVATE ${Python_INCLUDE_DIRS})
  target_link_libraries("brassboard_seq_${name}_mod" brassboard_seq_utils)
  set_target_properties("brassboard_seq_${name}_mod"
    PROPERTIES POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_NAME "${name}"
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
  install(TARGETS "brassboard_seq_${name}_mod"
    DESTINATION "${_PYTHON_SITE_INSTALL_DIR}/brassboard_seq")
endfunction()
bb_cxx_mod(action)
bb_cxx_mod(artiq_backend)
bb_cxx_mod(backend)
bb_cxx_mod(config)
bb_cxx_mod(event_time)
bb_cxx_mod(rfsoc_backend)
bb_cxx_mod(rtval)
bb_cxx_mod(scan)
bb_cxx_mod(seq)
bb_cxx_mod(utils)
bb_cxx_mod(yaml)

python_install_module("${CMAKE_CURRENT_SOURCE_DIR}/.." brassboard_seq
  __init__.py)
