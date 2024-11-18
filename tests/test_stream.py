#

import brassboard_seq_test_utils as test_utils

import pytest

def test_stream():
    io = test_utils.PyBytesStream()
    assert io.get_buf() == b''
    io.put(b'c'[0])
    assert io.get_buf() == b'c'
    assert io.get_buf() == b''
    io.flush()
    io.write("abcd")
    io.seek(10)
    io.write("321")
    io.seek(4)
    io.write("l9876k")
    assert io.get_buf() == b'abcdl9876k321'

    io.write('aaa')
    assert not io.fail()
    io.seek(-1)
    assert io.fail()
    io.clear()
    assert not io.fail()
    assert io.get_buf() == b'aaa'
    assert io.get_buf() == b''