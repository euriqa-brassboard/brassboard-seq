# cython: language_level=3

# Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>

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

from brassboard_seq.utils cimport _assume_not_none, assume_not_none, \
  PyErr_Format, PyExc_ValueError

cimport cython
from cpython cimport PyObject, PyDict_GetItemWithError, PyTuple_GET_SIZE

cdef tuple _translate_channel(Config self, tuple path):
    alias_cache = self.alias_cache
    cdef PyObject *resolvedp = PyDict_GetItemWithError(alias_cache, path)
    if resolvedp != NULL:
        return <tuple>resolvedp
    if PyTuple_GET_SIZE(path) > 10: # Hardcoded limit for loop detection
        name = '/'.join(path)
        PyErr_Format(PyExc_ValueError, 'Channel alias loop detected: %U',
                     <PyObject*>name)
    assume_not_none(path)
    cdef str prefix = path[0]
    _assume_not_none(<void*>self.channel_alias)
    if prefix in self.channel_alias:
        newpath = <tuple>(self.channel_alias[prefix]) + path[1:]
        resolved = _translate_channel(self, newpath)
        assume_not_none(alias_cache)
        alias_cache[path] = resolved
        return resolved
    _assume_not_none(<void*>self.supported_prefix)
    if not (prefix in self.supported_prefix):
        name = '/'.join(path)
        PyErr_Format(PyExc_ValueError, 'Unsupported channel name: %U',
                     <PyObject*>name)
    assume_not_none(alias_cache)
    alias_cache[path] = path
    return path

cdef tuple translate_channel(Config self, str name):
    assume_not_none(name)
    return _translate_channel(self, tuple(name.split('/')))

cdef class Config:
    def __cinit__(self):
        self.channel_alias = {}
        self.alias_cache = {}
        self.supported_prefix = set()

    def add_supported_prefix(self, str prefix, /):
        _assume_not_none(<void*>self.supported_prefix)
        self.supported_prefix.add(prefix)

    def add_channel_alias(self, str name, str target, /):
        assume_not_none(name)
        if '/' in name:
            PyErr_Format(PyExc_ValueError, 'Channel alias name may not contain "/"')
        _assume_not_none(<void*>self.alias_cache)
        self.alias_cache.clear()
        assume_not_none(target)
        self.channel_alias[name] = tuple(target.split('/'))

    def translate_channel(self, str name, /):
        return translate_channel(self, name)
