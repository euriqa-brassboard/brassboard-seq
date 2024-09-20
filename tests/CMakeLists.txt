#

cython_include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/brassboard_seq")

cython_link_libraries(brassboard_seq_utils)

python_test("${CMAKE_BINARY_DIR}"
  brassboard_seq_test_utils.pyx
  brassboard_seq_artiq_backend_utils.pyx
  brassboard_seq_rfsoc_backend_utils.pyx
  dummy_artiq.py
  dummy_pulse_compiler.py
  check_artiq_backend.py
  test_rtval.py # Keep this as the first in a parallel run since it takes the longest
  test_action.py
  test_artiq_backend.py
  test_artiq_backend_nodax.py
  test_backend.py
  test_config.py
  test_event_time.py
  test_rfsoc_backend.py
  test_scan.py
  test_seq.py
  test_seq_compile.py
  test_yaml.py
)
