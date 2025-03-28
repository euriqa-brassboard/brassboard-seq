import pytest

pytest.register_assert_rewrite("check_artiq_backend",
                               "py_test_utils",
                               "rfsoc_test_utils")
