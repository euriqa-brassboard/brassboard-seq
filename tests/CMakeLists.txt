#

cython_include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/brassboard_seq")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_definitions(-UNDEBUG)

cython_link_libraries(brassboard_seq_utils)

function(test_cxx_mod name)
  add_library("test_${name}_mod" MODULE "${name}.cpp")
  target_include_directories("test_${name}_mod" PRIVATE ${Python_INCLUDE_DIRS})
  target_link_libraries("test_${name}_mod" brassboard_seq_utils)
  set_target_properties("test_${name}_mod"
    PROPERTIES POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_NAME "${name}"
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}")
endfunction()
test_cxx_mod(brassboard_seq_test_utils)
test_cxx_mod(brassboard_seq_artiq_backend_utils)
test_cxx_mod(brassboard_seq_rfsoc_backend_utils)

python_test("${CMAKE_BINARY_DIR}"
  conftest.py
  dummy_artiq.py
  dummy_pulse_compiler.py
  check_artiq_backend.py
  py_test_utils.py
  rfsoc_test_utils.py
  # Keep these as the first in a parallel run since they take the longest
  test_max_range.py
  test_suffix_array.py
  test_rfsoc_jv1.py
  test_rfsoc_jv1_3.py
  test_bits.py
  test_rtval.py
  #
  test_action.py
  test_artiq_backend.py
  test_artiq_backend_nodax.py
  test_backend.py
  test_config.py
  test_event_time.py
  test_rfsoc_backend.py
  test_rfsoc_backend_jv1.py
  test_rfsoc_backend_jv1_3.py
  test_scan.py
  test_seq.py
  test_seq_compile.py
  test_stream.py
  test_utils.py
  test_yaml.py
)
