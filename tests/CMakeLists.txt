#

python_test("${CMAKE_BINARY_DIR}"
  test_rtval.py # Keep this as the first in a parallel run since it takes the longest
)
