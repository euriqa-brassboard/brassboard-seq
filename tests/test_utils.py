#

import py_test_utils as test_utils

import pytest
import random

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

def test_perm_alloc_iter():
    s = 0
    c = test_utils.IntCollector()
    assert c.sum() == 0
    for i in range(20000):
        v = random.randint(-10000, 10000)
        s += v
        c.add_int(v)
        assert c.sum() == s

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

def test_stringio():
    io = test_utils.IOBuff()
    assert io.getvalue() == ""

    io.write("aaa")
    io.write("bbb")
    io.write("ccc")
    io.write("ddd")
    assert io.getvalue() == "aaabbbcccddd"

    io.write_ascii(b"ee")
    assert io.getvalue() == "aaabbbcccdddee"
    io.write_rep_ascii(3, b"df")
    assert io.getvalue() == "aaabbbcccdddeedfdfdf"

    io.write("α")
    io.write("z")
    assert io.getvalue() == "aaabbbcccdddeedfdfdfαz"

    io.write("啊")
    assert io.getvalue() == "aaabbbcccdddeedfdfdfαz啊"

    io.write_ascii(b"mmm")
    assert io.getvalue() == "aaabbbcccdddeedfdfdfαz啊mmm"
    io.write_rep_ascii(0, b"adf")
    assert io.getvalue() == "aaabbbcccdddeedfdfdfαz啊mmm"
    io.write_rep_ascii(2, b"asdf")
    assert io.getvalue() == "aaabbbcccdddeedfdfdfαz啊mmmasdfasdf"
