# cython: language_level=3

# Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not,
# see <http://www.gnu.org/licenses/>.

# Do not use relative import since it messes up cython file name tracking
from brassboard_seq.utils cimport assume_not_none, PyDict_GET_SIZE

cdef np
import numpy as np

from cpython cimport PyList_GET_SIZE, PyList_GET_ITEM, PyTuple_GET_SIZE

from libcpp.vector cimport vector

cdef np_bool = np.bool_
cdef tuple np_num_types = (np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64,
                           np.float16, np.float32, np.float64, np.float128)
cdef np_array = np.ndarray

cdef inline bint needs_quote(str s) except -1:
    # Assuming string non-empty
    if s.startswith(' '):
        return True
    if s.endswith(' '):
        return True

    # Good enough for now........
    return '"' in s or ':' in s or '\n' in s or '\b' in s or '\\' in s

cdef str print_string(str s, int indent, int cur_indent):
    if not s:
        return '""'
    if needs_quote(s):
        s = s.replace('\\', '\\\\')
        s = s.replace('"', '\\"')
        s = s.replace('\n', '\\n')
        s = s.replace('\b', '\\b')
        s = '"' + s + '"'
    if indent < cur_indent and cur_indent + len(s) > 85:
        return '\n' + ' ' * indent + s
    return s

cdef str print_single_field_dict(obj, int indent, int cur_indent, str prefix_name=None):
    cdef list strary = []
    if prefix_name is not None:
        strary.append(prefix_name)
    while isinstance(obj, dict):
        if PyDict_GET_SIZE(obj) != 1:
            break
        for k, v in (<dict>obj).items():
            strary.append(k)
            obj = v
            break
    assert PyList_GET_SIZE(strary)
    name = '.'.join(strary) + ':'
    strfield = print_generic(obj, indent + 2, indent + len(name) + 1)
    if strfield and not strfield.startswith('\n'):
        s = name + ' ' + strfield
    else:
        s = name + strfield
    if indent < cur_indent:
        return '\n' + ' ' * indent + s
    return s

cdef str print_dict_field(k, v, int indent):
    cdef int keylen = len(k)
    if isinstance(v, dict) and PyDict_GET_SIZE(v) == 1:
        return print_single_field_dict(v, indent, indent, k)
    cdef int new_indent = indent + 2 + keylen
    s = print_generic(v, indent + 2, indent + 2 + keylen)
    if s.startswith('\n'):
        return k + ':' + s
    return k + ': ' + s

cdef str print_dict(dict obj, int indent, int cur_indent):
    assume_not_none(obj)
    cdef int nmembers = PyDict_GET_SIZE(obj)
    if nmembers == 0:
        return '{}'
    elif nmembers == 1:
        return print_single_field_dict(obj, indent, cur_indent)
    strary = [print_dict_field(k, v, indent) for k, v in obj.items()]
    s = ('\n' + ' ' * indent).join(strary)
    if indent < cur_indent:
        return '\n' + ' ' * indent + s
    return s

cdef str print_scalar(obj, int indent, int cur_indent):
    if isinstance(obj, bool) or isinstance(obj, np_bool):
        return "true" if obj else "false"
    elif (isinstance(obj, float) or isinstance(obj, int) or
          isinstance(obj, np_num_types)):
        return str(obj)
    elif isinstance(obj, str):
        return print_string(<str>obj, indent, cur_indent)
    elif isinstance(obj, dict):
        return print_dict(<dict>obj, indent, cur_indent)
    return f"<unknown object {obj}>"

cdef str print_array(ary, int indent, int cur_indent):
    cdef list strary = []
    cdef bint all_short_scalar = True
    cdef str s
    for v in ary:
        if isinstance(v, bool) or isinstance(v, np_bool):
            strary.append("true" if v else "false")
        elif (isinstance(v, float) or isinstance(v, int) or
              isinstance(v, np_num_types)):
            strary.append(str(v))
        elif isinstance(v, str):
            s = print_string(<str>v, 0, 0)
            if len(s) > 16:
                all_short_scalar = False
            strary.append(s)
        elif ((isinstance(v, list) and PyList_GET_SIZE(v) == 0) or
              (isinstance(v, tuple) and PyTuple_GET_SIZE(v) == 0) or
              (isinstance(v, np_array) and len(v) == 0)):
            strary.append("[]")
        elif isinstance(v, dict) and PyDict_GET_SIZE(v) == 0:
            strary.append("{}")
        else:
            strary.append(print_generic(v, indent + 2, indent + 2))
            all_short_scalar = False

    cdef int threshold
    cdef int linelen
    cdef int i
    cdef int nele
    if all_short_scalar:
        threshold = max(85 - cur_indent, 50)
        single_line = indent < cur_indent
        prefix = '['
        linelen = 0
        nele = PyList_GET_SIZE(strary)
        strary2 = []
        for i in range(nele):
            if linelen > threshold:
                if single_line:
                    single_line = False
                    threshold = max(85 - indent, 50)
                    prefix = '\n' + ' ' * indent + prefix
                if linelen > threshold and i < nele - 1:
                    # at least two elements on the last line
                    prefix += ', '.join(strary2) + ',\n' + ' ' * (indent + 1)
                    strary2.clear()
                    linelen = 0
            s = <str>PyList_GET_ITEM(strary, i)
            strary2.append(s)
            linelen += len(s) + 2
        return prefix + ', '.join(strary2) + ']'
    s = '- ' + ('\n' + ' ' * indent + '- ').join(strary)
    if indent < cur_indent:
        return '\n' + ' ' * indent + s
    return s

cdef str print_generic(obj, int indent, int cur_indent):
    if isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, np_array):
        return print_array(obj, indent, cur_indent)
    else:
        return print_scalar(obj, indent, cur_indent)

# Relaxed YAML-like printer
cpdef str sprint(obj, int indent=0):
    return print_generic(obj, indent, indent)
