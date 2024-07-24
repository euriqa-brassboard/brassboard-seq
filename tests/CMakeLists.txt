#

cython_include_directories("${CMAKE_BINARY_DIR}")
include_directories("${CMAKE_SOURCE_DIR}/brassboard_seq")

python_test("${CMAKE_BINARY_DIR}"
  brassboard_seq_test_utils.pyx
  test_rtval.py # Keep this as the first in a parallel run since it takes the longest
  test_action.py
  test_backend.py
  test_config.py
  test_event_time.py
  test_scan.py
  test_scan_no_yaml.py
  test_seq.py
  test_seq_compile.py
)
