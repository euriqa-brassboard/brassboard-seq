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
from cpython cimport PyTypeObject, PyFloat_AS_DOUBLE, PyTuple_GET_ITEM
from libc cimport math as cmath

cdef extern from "src/rtval.cpp" namespace "brassboard_seq::rtval":
    PyTypeObject *RTVal_Type
    object new_expr2_wrap1(ValueType, object, object) except +
    object build_addsub(object v0, object v1, bint) except +

RTVal_Type = <PyTypeObject*>RuntimeValue

cdef int operator_precedence(ValueType type_) noexcept:
    if type_ == ValueType.Add or type_ == ValueType.Sub:
        return 3
    if type_ == ValueType.Mul or type_ == ValueType.Div:
        return 2
    if (type_ == ValueType.CmpLT or type_ == ValueType.CmpGT or
        type_ == ValueType.CmpLE or type_ == ValueType.CmpGE):
        return 4
    if type_ == ValueType.CmpNE or type_ == ValueType.CmpEQ:
        return 5
    if type_ == ValueType.And:
        return 6
    if type_ == ValueType.Or:
        return 8
    if type_ == ValueType.Pow:
        return 7
    # if type_ == ValueType.Not: # Not printed as operator anymore
    #     return 1
    return 0

cdef bint needs_parenthesis(RuntimeValue v, ValueType parent_type) noexcept:
    cdef int op_self = operator_precedence(v.type_)
    cdef int op_parent = operator_precedence(parent_type)
    if op_self == 0 or op_parent == 0 or op_self < op_parent:
        return False
    if op_self > op_parent:
        return True
    if parent_type == ValueType.Add or parent_type == ValueType.Mul:
        return False
    return True

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
    if type_ == ValueType.Extern:
        cb = v.cb_arg2
        if isinstance(cb, ExternCallback):
            write(str(cb))
        else:
            write(f'extern({cb})')
    elif type_ == ValueType.ExternAge:
        cb = v.cb_arg2
        if isinstance(cb, ExternCallback):
            write(str(cb))
        else:
            write(f'extern_age({cb})')
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

cdef np_add = np.add
cdef np_subtract = np.subtract
cdef np_multiply = np.multiply
cdef np_divide = np.divide
cdef np_remainder = np.remainder
cdef np_bitwise_and = np.bitwise_and
cdef np_bitwise_or = np.bitwise_or
cdef np_bitwise_xor = np.bitwise_xor
cdef np_logical_not = np.logical_not
cdef np_power = np.power
cdef np_less = np.less
cdef np_greater = np.greater
cdef np_less_equal = np.less_equal
cdef np_greater_equal = np.greater_equal
cdef np_equal = np.equal
cdef np_not_equal = np.not_equal

cdef np_fmin = np.fmin
cdef np_fmax = np.fmax

cdef np_abs = np.abs
cdef np_ceil = np.ceil
cdef np_exp = np.exp
cdef np_expm1 = np.expm1
cdef np_floor = np.floor
cdef np_log = np.log
cdef np_log1p = np.log1p
cdef np_log2 = np.log2
cdef np_log10 = np.log10
cdef np_sqrt = np.sqrt
cdef np_arcsin = np.arcsin
cdef np_arccos = np.arccos
cdef np_arctan = np.arctan
cdef np_arctan2 = np.arctan2
cdef np_arcsinh = np.arcsinh
cdef np_arccosh = np.arccosh
cdef np_arctanh = np.arctanh
cdef np_sin = np.sin
cdef np_cos = np.cos
cdef np_tan = np.tan
cdef np_sinh = np.sinh
cdef np_cosh = np.cosh
cdef np_tanh = np.tanh
cdef np_hypot = np.hypot
cdef np_rint = np.rint

def get_value(v, age):
    cdef py_object pyage
    if isinstance(age, int):
        pyage.set_obj(age)
    if is_rtval(v):
        rt_eval_cache(<RuntimeValue>v, age, pyage)
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

    def eval(self, age, /):
        cdef py_object pyage
        if isinstance(age, int):
            pyage.set_obj(age)
        rt_eval_cache(self, <unsigned>age, pyage)
        return rtval_cache(self).to_py()

    def __str__(self):
        io = StringIO()
        show(io, io.write, self)
        return io.getvalue()

    def __repr__(self):
        io = StringIO()
        show(io, io.write, self)
        return io.getvalue()

    # It's too easy to accidentally use this in control flow/assertion
    def __bool__(self):
        PyErr_Format(PyExc_TypeError, "Cannot convert runtime value to boolean")

    def __add__(self, other):
        return build_addsub(self, other, False)
    def __sub__(self, other):
        return build_addsub(self, other, True)

    def __mul__(self, other):
        return new_expr2_wrap1(ValueType.Mul, self, other)

    def __truediv__(self, other):
        return new_expr2_wrap1(ValueType.Div, self, other)

    def __and__(self, other):
        return new_expr2_wrap1(ValueType.And, self, other)

    def __or__(self, other):
        return new_expr2_wrap1(ValueType.Or, self, other)

    def __xor__(self, other):
        return new_expr2_wrap1(ValueType.Xor, self, other)

    def __pow__(self, other):
        return new_expr2_wrap1(ValueType.Pow, self, other)

    def __mod__(self, other):
        return new_expr2_wrap1(ValueType.Mod, self, other)

    def __pos__(self):
        return self
    def __neg__(self):
        return build_addsub(0, self, True)

    def __richcmp__(self, other, int op):
        typ = pycmp2valcmp(op)
        cdef RuntimeValue v2
        if is_rtval(other):
            if self is other:
                return (typ == ValueType.CmpLE or typ == ValueType.CmpGE or
                        typ == ValueType.CmpEQ)
            v2 = <RuntimeValue>other
        else:
            v2 = new_const(other, <RuntimeValue>None)
        return new_expr2(typ, self, v2)

    def __abs__(self):
        if self.type_ == ValueType.Abs:
            return self
        return new_expr1(ValueType.Abs, self)

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

    def __array_ufunc__(self, ufunc, methods, /, *inputs):
        if methods != '__call__':
            return <object>Py_NotImplemented
        # Needed for numpy type support
        if ufunc is np_add:
            return build_addsub(<object>PyTuple_GET_ITEM(inputs, 0),
                                <object>PyTuple_GET_ITEM(inputs, 1), False)
        if ufunc is np_subtract:
            return build_addsub(<object>PyTuple_GET_ITEM(inputs, 0),
                                <object>PyTuple_GET_ITEM(inputs, 1), True)
        if ufunc is np_multiply:
            return new_expr2_wrap1(ValueType.Mul, <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_divide:
            return new_expr2_wrap1(ValueType.Div, <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_remainder:
            return new_expr2_wrap1(ValueType.Mod, <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_bitwise_and:
            return new_expr2_wrap1(ValueType.And, <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_bitwise_or:
            return new_expr2_wrap1(ValueType.Or, <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_bitwise_xor:
            return new_expr2_wrap1(ValueType.Xor, <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_logical_not:
            if self.type_ == ValueType.Not:
                return rt_convert_bool(self.arg0)
            return new_expr1(ValueType.Not, self)
        if ufunc is np_power:
            return new_expr2_wrap1(ValueType.Pow, <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_less:
            return new_expr2_wrap1(ValueType.CmpLT,
                                   <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_greater:
            return new_expr2_wrap1(ValueType.CmpGT,
                                   <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_less_equal:
            return new_expr2_wrap1(ValueType.CmpLE,
                                   <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_greater_equal:
            return new_expr2_wrap1(ValueType.CmpGE,
                                   <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_equal:
            return new_expr2_wrap1(ValueType.CmpEQ,
                                   <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_not_equal:
            return new_expr2_wrap1(ValueType.CmpNE,
                                   <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_fmin:
            v1 = PyTuple_GET_ITEM(inputs, 0)
            v2 = PyTuple_GET_ITEM(inputs, 1)
            if v1 == v2:
                return <object>v1
            return new_expr2_wrap1(ValueType.Min, <object>v1, <object>v2)
        if ufunc is np_fmax:
            v1 = PyTuple_GET_ITEM(inputs, 0)
            v2 = PyTuple_GET_ITEM(inputs, 1)
            if v1 == v2:
                return <object>v1
            return new_expr2_wrap1(ValueType.Max, <object>v1, <object>v2)
        if ufunc is np_abs:
            if self.type_ == ValueType.Abs:
                return self
            return new_expr1(ValueType.Abs, self)
        if ufunc is np_ceil:
            if self.type_ == ValueType.Ceil:
                return self
            return new_expr1(ValueType.Ceil, self)
        if ufunc is np_exp:
            return new_expr1(ValueType.Exp, self)
        if ufunc is np_expm1:
            return new_expr1(ValueType.Expm1, self)
        if ufunc is np_floor:
            if self.type_ == ValueType.Floor:
                return self
            return new_expr1(ValueType.Floor, self)
        if ufunc is np_log:
            return new_expr1(ValueType.Log, self)
        if ufunc is np_log1p:
            return new_expr1(ValueType.Log1p, self)
        if ufunc is np_log2:
            return new_expr1(ValueType.Log2, self)
        if ufunc is np_log10:
            return new_expr1(ValueType.Log10, self)
        if ufunc is np_sqrt:
            return new_expr1(ValueType.Sqrt, self)
        if ufunc is np_arcsin:
            return new_expr1(ValueType.Asin, self)
        if ufunc is np_arccos:
            return new_expr1(ValueType.Acos, self)
        if ufunc is np_arctan:
            return new_expr1(ValueType.Atan, self)
        if ufunc is np_arctan2:
            return new_expr2_wrap1(ValueType.Atan2,
                                   <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_arcsinh:
            return new_expr1(ValueType.Asinh, self)
        if ufunc is np_arccosh:
            return new_expr1(ValueType.Acosh, self)
        if ufunc is np_arctanh:
            return new_expr1(ValueType.Atanh, self)
        if ufunc is np_sin:
            return new_expr1(ValueType.Sin, self)
        if ufunc is np_cos:
            return new_expr1(ValueType.Cos, self)
        if ufunc is np_tan:
            return new_expr1(ValueType.Tan, self)
        if ufunc is np_sinh:
            return new_expr1(ValueType.Sinh, self)
        if ufunc is np_cosh:
            return new_expr1(ValueType.Cosh, self)
        if ufunc is np_tanh:
            return new_expr1(ValueType.Tanh, self)
        if ufunc is np_hypot:
            return new_expr2_wrap1(ValueType.Hypot,
                                   <object>PyTuple_GET_ITEM(inputs, 0),
                                   <object>PyTuple_GET_ITEM(inputs, 1))
        if ufunc is np_rint:
            if self.type_ == ValueType.Rint:
                return self
            return new_expr1(ValueType.Rint, self)
        return <object>Py_NotImplemented

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

def round_int64(_v, /):
    cdef RuntimeValue v
    if is_rtval(_v):
        return rt_round_int64(<RuntimeValue>_v)
    if isinstance(_v, cnpy.ndarray):
        ary = <cnpy.ndarray>_v
        if cnpy.PyArray_TYPE(ary) == cnpy.NPY_INT64:
            return ary
        if not (cnpy.PyArray_ISINTEGER(ary) or cnpy.PyArray_ISBOOL(ary)):
            ary = <cnpy.ndarray>np_rint(ary)
        return cnpy.PyArray_Cast(ary, cnpy.NPY_INT64)
    if type(_v) is int:
        return _v
    if type(_v) is float:
        return cmath.llrint(PyFloat_AS_DOUBLE(_v))
    return round(_v)

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

    def __call__(self, age, /):
        _v = getattr(self.obj, self.fieldname)
        if not is_rtval(_v):
            return _v
        cdef RuntimeValue v = <RuntimeValue>_v
        if (v.type_ == ValueType.ExternAge and v.cb_arg2 is self):
            PyErr_Format(PyExc_ValueError, 'RT property have not been assigned.')
        cdef py_object pyage
        pyage.set_obj(age)
        rt_eval_cache(v, <unsigned>age, pyage)
        return rtval_cache(v).to_py()

cdef rtprop_callback new_rtprop_callback(obj, str fieldname):
    self = <rtprop_callback>rtprop_callback.__new__(rtprop_callback)
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
        value = new_extern_age(new_rtprop_callback(obj, fieldname))
        setattr(obj, fieldname, value)
        return value
