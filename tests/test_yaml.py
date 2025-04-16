#

from brassboard_seq.yaml import sprint as yaml_print
from py_test_utils import yaml_io_print

import numpy as np

import pytest

class ObjType:
    pass

@pytest.mark.parametrize("yaml_print", [yaml_print, yaml_io_print])
def test_print(yaml_print):
    assert yaml_print(False) == 'false'
    assert yaml_print(True) == 'true'
    assert yaml_print(1) == '1'
    assert yaml_print(1.0) == '1.0'
    assert yaml_print(np.int32(1)) == '1'
    assert yaml_print(np.float64(1.0)) == '1.0'
    assert yaml_print("") == '""'
    assert yaml_print("a") == 'a'
    assert yaml_print("a\"") == '\"a\\\"\"'
    assert yaml_print("a\n") == '\"a\\n\"'
    assert yaml_print("a\b") == '\"a\\b\"'
    assert yaml_print("a\\x") == '\"a\\\\x\"'
    assert yaml_print("a 1") == 'a 1'
    assert yaml_print(" 1") == '" 1"'
    assert yaml_print("1 ") == '"1 "'
    assert yaml_print({}) == '{}'
    assert yaml_print([]) == '[]'
    assert yaml_print(()) == '[]'
    assert yaml_print((True, 1, 2.3, "a", {}, np.bool_(False))) == '[true, 1, 2.3, a, {}, false]'
    obj = ObjType()
    assert yaml_print(obj) == f"<unknown object {obj}>"

    assert (yaml_print(dict(a12345=dict(bcd=dict(xyz12345k=dict(zzzxxx="aa" * 30))))) ==
            "a12345.bcd.xyz12345k.zzzxxx:\n  " + "aa" * 30)
    assert yaml_print([1, 2.3, [], "adf", {}, np.array([]), (),
                       False]) == '[1, 2.3, [], adf, {}, [], [], false]'
    assert yaml_print(["xxx123456kkkklllllasdfasdf",
                       2]) == '- xxx123456kkkklllllasdfasdf\n- 2'
    assert yaml_print([[1, 2.0], 2]) == '- [1, 2.0]\n- 2'
    assert yaml_print(dict(a=3, e=dict(b=dict(c=dict(a=2, b=3))), f="abc")) == "a: 3\ne.b.c:\n  a: 2\n  b: 3\nf: abc"
    assert yaml_print([dict(a=dict(d=3))]) == "- a.d: 3"
    assert yaml_print(dict(a=dict(xyzq=[1, 2, 3, 4, 5]))) == "a.xyzq: [1, 2, 3, 4, 5]"
    assert (yaml_print(dict(a1234=dict(xyzq=["123456", "abcdef", 1, 2, 3, 4, 5,
                                             6, 7, 8, 9, 10, [], 2., 3., 4.,
                                             1, 2, 3]))) ==
            "a1234.xyzq: [123456, abcdef, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [], 2.0, 3.0, 4.0, 1, 2, 3]")
    assert (yaml_print(dict(a1234=dict(xyzq=["123456", "abcdef", 1, 2, 3, 4, 5,
                                             6, 7, 8, 9, 10, [], 2., 3., 4.,
                                             1, 2, 3, 4]))) ==
            "a1234.xyzq:\n  [123456, abcdef, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [], 2.0, 3.0, 4.0, 1, 2, 3, 4]")
    assert (yaml_print(dict(a1234=dict(xyzq=["123456", "abcdef", 1, 2, 3, 4, 5,
                                             6, 7, 8, 9, 10, [], 2., 3., 4.,
                                             1, 2, 3, 4, 5, 6, 7]))) ==
            "a1234.xyzq:\n  [123456, abcdef, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [], 2.0, 3.0, 4.0, 1, 2, 3, 4, 5, 6, 7]")
    assert (yaml_print(dict(a1234=dict(xyzq=["123456", "abcdef", 1, 2, 3, 4, 5,
                                             6, 7, 8, 9, 10, [], 2., 3., 4.,
                                             1, 2, 3, 4, 5, 6, 7, 8]))) ==
            "a1234.xyzq:\n  [123456, abcdef, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [], 2.0, 3.0, 4.0, 1, 2, 3, 4, 5, 6,\n   7, 8]")
    assert (yaml_print([["123456", "abcdef", 1, 2, 3, 4, 5,
                                             6, 7, 8, 9, 10, [], 2., 3., 4.,
                                             1, 2, 3, 4, 5, 6, 7]]) ==
            "- [123456, abcdef, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [], 2.0, 3.0, 4.0, 1, 2, 3, 4, 5, 6, 7]")
    assert (yaml_print([["123456", "abcdef", 1, 2, 3, 4, 5,
                                             6, 7, 8, 9, 10, [], 2., 3., 4.,
                                             1, 2, 3, 4, 5, 6, 7, 8]]) ==
            "- [123456, abcdef, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [], 2.0, 3.0, 4.0, 1, 2, 3, 4, 5, 6,\n   7, 8]")

    assert yaml_print(dict(abcd="a" * 70)) == 'abcd: ' + 'a' * 70
    assert yaml_print(dict(abcd="a" * 90)) == 'abcd:\n  ' + 'a' * 90
    assert yaml_print(dict(abcd=[[1, 2], [3, 4]])) == 'abcd:\n  - [1, 2]\n  - [3, 4]'

    assert yaml_print(dict(x=dict(y=False), a=dict(x=2, y=3)), 2) == """x.y: false
  a:
    x: 2
    y: 3"""

    assert yaml_print(np.array([True, False, True, True])) == "[true, false, true, true]"
    assert yaml_print(np.array([1, 2, 3, 10, 9, -3])) == "[1, 2, 3, 10, 9, -3]"
    assert yaml_print(np.array([1.3, 2.4, 3.0, 9.8, -9.3, -3.3])) == "[1.3, 2.4, 3.0, 9.8, -9.3, -3.3]"

    with pytest.raises(TypeError, match="yaml dict key must be str"):
        yaml_print({1: 2})

    with pytest.raises(TypeError, match="yaml only support ndarray of dimension 1"):
        yaml_print(np.array([[1, 2], [3, 4]]))

    with pytest.raises(TypeError, match="print\\(\\) takes at most 2 positional arguments \\(3 given\\)"):
        yaml_print([], 1, 2)

    with pytest.raises(TypeError, match="Unexpected type .* for indent"):
        yaml_print([], "aa")

    with pytest.raises(TypeError, match="indent cannot be negative"):
        yaml_print([], -2)
