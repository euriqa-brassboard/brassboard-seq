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

# Do not use relative import since it messes up cython file name tracking

from cpython cimport PyObject

cdef extern from * namespace "brassboard_seq::event_time":
    PyObject *py_time_scale()

# Manually set the field since I can't make cython automatically do this
# without also declaring the c struct again...
globals()['EventTime'] = EventTime

def time_scale():
    return <object>py_time_scale()
