#

import brassboard_seq_test_utils as test_utils

import pytest

@pytest.mark.parametrize("streamm_type",
                         [test_utils.PyBytesStream, test_utils.PyByteArrayStream])
def test_stream(streamm_type):
    io = streamm_type()
    assert io.get_buf() == b''
    io.put(b'c'[0])
    assert io.get_buf() == b'c'
    assert io.get_buf() == b''
    io.flush()
    io.write("abcd")
    io.seek(10)
    io.write("321")
    io.seek(4, 'beg')
    io.write("l9876k")
    assert io.get_buf() == b'abcdl9876k321'

    io.write('aaa')
    assert not io.fail()
    io.seek(-1)
    assert io.fail()
    io.clear()
    assert not io.fail()
    io.seek(0, 'cur')
    assert not io.fail()
    io.seek(-1, 'beg')
    assert io.fail()
    io.clear()
    assert not io.fail()
    io.seek(1, 'end')
    assert io.fail()
    io.clear()
    assert not io.fail()
    assert io.get_buf() == b'aaa'
    assert io.get_buf() == b''

    io.write('aaa')
    assert not io.fail()
    io.seek(-2, 'cur')
    io.write('bbbb')
    io.seek(-2, 'end')
    io.seek(-2, 'cur')
    io.write('c')
    assert not io.fail()
    assert io.get_buf() == b'acbbb'

def test_istream():
    assert test_utils.test_istream_seek(0)
    assert test_utils.test_istream_seek(0, 'beg')
    assert test_utils.test_istream_seek(0, 'end')
    assert test_utils.test_istream_seek(0, 'cur')

    assert test_utils.test_istream_ba_seek(0)
    assert test_utils.test_istream_ba_seek(0, 'beg')
    assert test_utils.test_istream_ba_seek(0, 'end')
    assert test_utils.test_istream_ba_seek(0, 'cur')
