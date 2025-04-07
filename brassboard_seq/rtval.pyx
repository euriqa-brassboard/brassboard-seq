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
from brassboard_seq.utils cimport PyErr_Format, PyExc_ValueError

# Manually set the field since I can't make cython automatically do this
# without also declaring the c struct again...
globals()['RuntimeValue'] = RuntimeValue

cimport cython

cdef extern from "src/_rtprop.cpp" namespace "brassboard_seq::rtval":
    TagVal rtprop_callback_func(rtprop_callback self, unsigned age) except +
    composite_rtprop_data get_composite_rtprop_data(CompositeRTProp prop, object obj,
                                                    object, composite_rtprop_data) except +
    object composite_rtprop_get_res(CompositeRTProp self, object obj,
                                    object, composite_rtprop_data) except +
    void assert_layout_compatible(ExternCallback)

assert_layout_compatible(None)

def get_value(v, unsigned age):
    if is_rtval(v):
        rt_eval_cache(<RuntimeValue>v, age)
        return rtval_cache(<RuntimeValue>v).to_py()
    return v

def inv(v, /):
    if type(v) is bool:
        return v is False
    cdef RuntimeValue _v
    if is_rtval(v):
        _v = <RuntimeValue>v
        if _v.type_ == ValueType.Not:
            return rt_convert_bool(_v.arg0)
        return new_expr1(ValueType.Not, _v)
    return not v

def convert_bool(_v):
    if is_rtval(_v):
        return rt_convert_bool(<RuntimeValue>_v)
    return bool(_v)

def ifelse(b, v1, v2):
    if rt_same_value(v1, v2):
        return v1
    if is_rtval(b):
        return new_select(<RuntimeValue>b, v1, v2)
    return v1 if b else v2

def same_value(v1, v2):
    return rt_same_value(v1, v2)

@cython.auto_pickle(False)
cdef class ExternCallback:
    pass

cdef str rtprop_prefix = '_RTProp_value_'
cdef int rtprop_prefix_len = len(rtprop_prefix)

@cython.internal
@cython.auto_pickle(False)
@cython.final
cdef class rtprop_callback(ExternCallback):
    cdef obj
    cdef str fieldname

    def __str__(self):
        name = self.fieldname[rtprop_prefix_len:]
        return f'<RTProp {name} for {self.obj}>'

cdef rtprop_callback new_rtprop_callback(obj, str fieldname):
    self = <rtprop_callback>rtprop_callback.__new__(rtprop_callback)
    self.fptr = <void*><TagVal(*)(rtprop_callback, unsigned)>rtprop_callback_func
    self.obj = obj
    self.fieldname = fieldname
    return self

@cython.final
cdef class RTProp:
    cdef str fieldname

    def get_state(self, obj):
        try:
            return getattr(obj, self.fieldname)
        except AttributeError:
            return

    def set_state(self, obj, val):
        if val is None:
            delattr(obj, self.fieldname)
        else:
            setattr(obj, self.fieldname, val)

    def __set_name__(self, owner, name):
        self.fieldname = rtprop_prefix + name

    def __set__(self, obj, value):
        if self.fieldname is None:
            PyErr_Format(PyExc_ValueError, 'Cannot determine runtime property name')
        setattr(obj, self.fieldname, value)

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        if self.fieldname is None:
            PyErr_Format(PyExc_ValueError, 'Cannot determine runtime property name')
        fieldname = self.fieldname
        try:
            return getattr(obj, fieldname)
        except AttributeError:
            pass
        value = new_extern_age(new_rtprop_callback(obj, fieldname), float)
        setattr(obj, fieldname, value)
        return value

@cython.final
@cython.internal
cdef class composite_rtprop_data:
    cdef object ovr
    cdef object cache
    cdef uint8_t compiled
    cdef uint8_t filled

@cython.final
cdef class CompositeRTProp:
    cdef str fieldname
    cdef object cb

    def __init__(self, cb):
        self.cb = cb

    def get_state(self, obj):
        return get_composite_rtprop_data(self, obj, composite_rtprop_data, None).ovr

    def set_state(self, obj, val):
        get_composite_rtprop_data(self, obj, composite_rtprop_data, None).ovr = val

    def __set_name__(self, owner, name):
        self.fieldname = '__CompositeRTProp__' + name

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        return composite_rtprop_get_res(self, obj, composite_rtprop_data, None)
