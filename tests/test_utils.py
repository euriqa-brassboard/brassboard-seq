#

import brassboard_seq_test_utils as test_utils

import pytest

def test_to_chars():
    assert test_utils.int_to_chars(123) == b'123'
    assert test_utils.int_to_chars(12333) == b'12333'
    with pytest.raises(RuntimeError):
        test_utils.int_to_chars(123330)
    with pytest.raises(RuntimeError):
        test_utils.int_to_chars(-12333)
