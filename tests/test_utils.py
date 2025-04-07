#

import py_test_utils as test_utils

import pytest

def test_to_chars():
    assert test_utils.int_to_chars(123) == b'123'
    assert test_utils.int_to_chars(12333) == b'12333'
    with pytest.raises(RuntimeError):
        test_utils.int_to_chars(123330)
    with pytest.raises(RuntimeError):
        test_utils.int_to_chars(-12333)

def test_cpp_throw():
    assert test_utils.int_throw_if(0) == 0
    assert test_utils.int_throw_if_not(10) == 10
    with pytest.raises(RuntimeError):
        test_utils.int_throw_if(2)
    with pytest.raises(RuntimeError):
        test_utils.int_throw_if_not(0)

def test_check_num_arg():
    test_utils.check_num_arg(b"aaa", 0, 0, 4)
    test_utils.check_num_arg(b"aaa", 4, 0, 4)

    with pytest.raises(TypeError, match="xasdkfj\\(\\) takes at least 1 positional argument \\(0 given\\)"):
        test_utils.check_num_arg(b"xasdkfj", 0, 1, 4)
    with pytest.raises(TypeError, match="j98sdjfaskdj\\(\\) takes at most 4 positional arguments \\(5 given\\)"):
        test_utils.check_num_arg(b"j98sdjfaskdj", 5, 1, 4)
    with pytest.raises(TypeError, match="jowie8uasdf\\(\\) takes exactly 2 positional arguments \\(1 given\\)"):
        test_utils.check_num_arg(b"jowie8uasdf", 1, 2, 2)
    with pytest.raises(TypeError, match="jajsd89iwuer89\\(\\) takes exactly 2 positional arguments \\(3 given\\)"):
        test_utils.check_num_arg(b"jajsd89iwuer89", 3, 2, 2)
