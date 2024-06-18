#

cython_include_directories("${CMAKE_BINARY_DIR}")

python_test("${CMAKE_BINARY_DIR}"
  brassboard_seq_test_utils.pyx
  test_rtval.py # Keep this as the first in a parallel run since it takes the longest
)
