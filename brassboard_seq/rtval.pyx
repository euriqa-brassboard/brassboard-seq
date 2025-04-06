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
from brassboard_seq.utils cimport PyErr_Format, Py_NotImplemented, \
  PyExc_TypeError, PyExc_ValueError, init_library

cdef StringIO, np # hide import
from io import StringIO
import numpy as np
cimport numpy as cnpy
cnpy._import_array()

init_library()

cimport cython
from cpython cimport PyObject, PyTuple_GET_ITEM

cdef extern from "src/rtval.cpp" namespace "brassboard_seq::rtval":
    PyObject *RTVal_Type
    object new_expr2_wrap1(ValueType, object, object) except +
    object build_addsub(object v0, object v1, bint) except +
    bint needs_parenthesis(RuntimeValue v, ValueType parent_type)
    void update_rtvalue()
    TagVal rtprop_callback_func(rtprop_callback self, unsigned age) except +
    composite_rtprop_data get_composite_rtprop_data(CompositeRTProp prop, object obj,
                                                    object, composite_rtprop_data) except +
    object composite_rtprop_get_res(CompositeRTProp self, object obj,
                                    object, composite_rtprop_data) except +
    void assert_layout_compatible(RuntimeValue, ExternCallback)

RTVal_Type = <PyObject*>RuntimeValue
assert_layout_compatible(None, None)
update_rtvalue()

cdef int show_arg(io, write, RuntimeValue v, ValueType parent_type) except -1:
    cdef bint p = needs_parenthesis(v, parent_type)
    if p:
        write('(')
    show(io, write, v)
    if p:
        write(')')

cdef int show_binary(io, write, RuntimeValue v, str op, ValueType type_) except -1:
    show_arg(io, write, v.arg0, type_)
    write(op)
    show_arg(io, write, v.arg1, type_)

cdef int show_call1(io, write, RuntimeValue v, str f) except -1:
    write(f)
    write('(')
    show(io, write, v.arg0)
    write(')')

cdef int show_call2(io, write, RuntimeValue v, str f) except -1:
    write(f)
    write('(')
    show(io, write, v.arg0)
    write(', ')
    show(io, write, v.arg1)
    write(')')

cdef int show_call3(io, write, RuntimeValue v, str f) except -1:
    write(f)
    write('(')
    show(io, write, v.arg0)
    write(', ')
    show(io, write, v.arg1)
    write(', ')
    show(io, write, <RuntimeValue>v.cb_arg2)
    write(')')

cdef int show(io, write, RuntimeValue v) except -1:
    cdef ValueType type_ = v.type_
    if type_ == ValueType.Extern or type_ == ValueType.ExternAge:
        write(str(<ExternCallback>v.cb_arg2))
    elif type_ == ValueType.Arg:
        write(f'arg({v.cb_arg2})')
    elif type_ == ValueType.Const:
        print(rtval_cache(v).to_py(), end='', file=io)
    elif type_ == ValueType.Add:
        show_binary(io, write, v, ' + ', type_)
    elif type_ == ValueType.Sub:
        show_binary(io, write, v, ' - ', type_)
    elif type_ == ValueType.Mul:
        show_binary(io, write, v, ' * ', type_)
    elif type_ == ValueType.Div:
        show_binary(io, write, v, ' / ', type_)
    elif type_ == ValueType.CmpLT:
        show_binary(io, write, v, ' < ', type_)
    elif type_ == ValueType.CmpGT:
        show_binary(io, write, v, ' > ', type_)
    elif type_ == ValueType.CmpLE:
        show_binary(io, write, v, ' <= ', type_)
    elif type_ == ValueType.CmpGE:
        show_binary(io, write, v, ' >= ', type_)
    elif type_ == ValueType.CmpEQ:
        show_binary(io, write, v, ' == ', type_)
    elif type_ == ValueType.CmpNE:
        show_binary(io, write, v, ' != ', type_)
    elif type_ == ValueType.And:
        show_binary(io, write, v, ' & ', type_)
    elif type_ == ValueType.Or:
        show_binary(io, write, v, ' | ', type_)
    elif type_ == ValueType.Xor:
        show_binary(io, write, v, ' ^ ', type_)
    elif type_ == ValueType.Mod:
        show_binary(io, write, v, ' % ', type_)
    elif type_ == ValueType.Pow:
        show_binary(io, write, v, '**', type_)
    elif type_ == ValueType.Not:
        show_call1(io, write, v, 'inv')
    elif type_ == ValueType.Abs:
        show_call1(io, write, v, 'abs')
    elif type_ == ValueType.Ceil:
        show_call1(io, write, v, 'ceil')
    elif type_ == ValueType.Exp:
        show_call1(io, write, v, 'exp')
    elif type_ == ValueType.Expm1:
        show_call1(io, write, v, 'expm1')
    elif type_ == ValueType.Floor:
        show_call1(io, write, v, 'floor')
    elif type_ == ValueType.Log:
        show_call1(io, write, v, 'log')
    elif type_ == ValueType.Log1p:
        show_call1(io, write, v, 'log1p')
    elif type_ == ValueType.Log2:
        show_call1(io, write, v, 'log2')
    elif type_ == ValueType.Log10:
        show_call1(io, write, v, 'log10')
    elif type_ == ValueType.Sqrt:
        show_call1(io, write, v, 'sqrt')
    elif type_ == ValueType.Asin:
        show_call1(io, write, v, 'arcsin')
    elif type_ == ValueType.Acos:
        show_call1(io, write, v, 'arccos')
    elif type_ == ValueType.Atan:
        show_call1(io, write, v, 'arctan')
    elif type_ == ValueType.Asinh:
        show_call1(io, write, v, 'arcsinh')
    elif type_ == ValueType.Acosh:
        show_call1(io, write, v, 'arccosh')
    elif type_ == ValueType.Atanh:
        show_call1(io, write, v, 'arctanh')
    elif type_ == ValueType.Sin:
        show_call1(io, write, v, 'sin')
    elif type_ == ValueType.Cos:
        show_call1(io, write, v, 'cos')
    elif type_ == ValueType.Tan:
        show_call1(io, write, v, 'tan')
    elif type_ == ValueType.Sinh:
        show_call1(io, write, v, 'sinh')
    elif type_ == ValueType.Cosh:
        show_call1(io, write, v, 'cosh')
    elif type_ == ValueType.Tanh:
        show_call1(io, write, v, 'tanh')
    elif type_ == ValueType.Rint:
        show_call1(io, write, v, 'rint')
    elif type_ == ValueType.Max:
        show_call2(io, write, v, 'max')
    elif type_ == ValueType.Min:
        show_call2(io, write, v, 'min')
    elif type_ == ValueType.Int64:
        show_call1(io, write, v, 'int64')
    elif type_ == ValueType.Bool:
        show_call1(io, write, v, 'bool')
    elif type_ == ValueType.Atan2:
        show_call2(io, write, v, 'arctan2')
    elif type_ == ValueType.Hypot:
        show_call2(io, write, v, 'hypot')
    elif type_ == ValueType.Select:
        show_call3(io, write, v, 'ifelse')
    else:
        write('Unknown value')

def get_value(v, unsigned age):
    if is_rtval(v):
        rt_eval_cache(<RuntimeValue>v, age)
        return rtval_cache(<RuntimeValue>v).to_py()
    return v

@cython.auto_pickle(False)
@cython.c_api_binop_methods(True)
@cython.final
cdef class RuntimeValue:
    def __init__(self):
        # All instances should be constructed within cython code via
        # `RuntimeValue.__new__` or its wrapper.
        PyErr_Format(PyExc_TypeError, "RuntimeValue cannot be created directly")

    def eval(self, unsigned age, /):
        rt_eval_cache(self, age)
        return rtval_cache(self).to_py()

    def __str__(self):
        io = StringIO()
        show(io, io.write, self)
        return io.getvalue()

    def __repr__(self):
        io = StringIO()
        show(io, io.write, self)
        return io.getvalue()

    def __ceil__(self):
        if self.type_ == ValueType.Ceil:
            return self
        return new_expr1(ValueType.Ceil, self)

    def __floor__(self):
        if self.type_ == ValueType.Floor:
            return self
        return new_expr1(ValueType.Floor, self)

    def __round__(self):
        return rt_round_int64(self)

cdef np_logical_not = np.logical_not
def inv(v, /):
    if type(v) is bool:
        return v is False
    cdef RuntimeValue _v
    if is_rtval(v):
        _v = <RuntimeValue>v
        if _v.type_ == ValueType.Not:
            return rt_convert_bool(_v.arg0)
        return new_expr1(ValueType.Not, _v)
    if isinstance(v, cnpy.ndarray):
        return np_logical_not(v)
    return not v

def convert_bool(_v):
    if is_rtval(_v):
        return rt_convert_bool(<RuntimeValue>_v)
    if isinstance(_v, cnpy.ndarray):
        return cnpy.PyArray_Cast(_v, cnpy.NPY_BOOL)
    return bool(_v)

def ifelse(b, v1, v2):
    if (isinstance(b, cnpy.ndarray) or isinstance(v1, cnpy.ndarray) or
        isinstance(v2, cnpy.ndarray)):
        return cnpy.PyArray_Where(b, v1, v2)
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
