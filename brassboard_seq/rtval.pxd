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
from brassboard_seq.utils cimport py_object

from libc.stdint cimport *
from libcpp.vector cimport vector

cdef extern from "src/rtval.h" namespace "brassboard_seq::rtval":
    enum ValueType:
        Arg
        ExternAge
        Const
        Extern

        Add
        Sub
        Mul
        Div
        CmpLT
        CmpGT
        CmpLE
        CmpGE
        CmpNE
        CmpEQ
        And
        Or
        Xor
        Not
        Abs
        Ceil
        Exp
        Expm1
        Floor
        Log
        Log1p
        Log2
        Log10
        Pow
        Sqrt
        Asin
        Acos
        Atan
        Atan2
        Asinh
        Acosh
        Atanh
        Sin
        Cos
        Tan
        Sinh
        Cosh
        Tanh
        Hypot
        # Erf
        # Erfc
        # Gamma
        # Lgamma
        Rint
        Max
        Min
        Mod
        # Interp
        Select
        # Identity
        Int64
        Bool

    ValueType pycmp2valcmp(int op) noexcept

    enum class DataType(uint8_t):
        Bool
        Int64
        Float64
    DataType promote_type(DataType t1, DataType t2)
    DataType pytype_to_datatype(object) except +

    union GenVal:
        bint b_val
        int64_t i64_val
        double f64_val

    enum class EvalError(uint8_t):
        pass
    void throw_py_error(EvalError) except +
    void throw_py_error(EvalError, uintptr_t) except +

    cppclass TagVal:
        DataType type
        EvalError err
        GenVal val
        TagVal()
        TagVal(double)
        T get[T]()
        @staticmethod
        TagVal from_py(object obj) except +
        object to_py() except +
        bint is_zero()

    cppclass InterpFunction:
        void set_value(RuntimeValue, vector[DataType]&) except +
        void eval_all(unsigned, py_object&, RuntimeValue) except +
        TagVal call()

cdef class RuntimeValue:
    cdef ValueType type_
    cdef unsigned age
    cdef RuntimeValue arg0
    cdef RuntimeValue arg1
    cdef object cb_arg2 # Also used as argument index
    cdef TagVal cache

cdef int rt_eval_tagval(RuntimeValue self, unsigned age, py_object &pyage) except -1
cdef int interp_function_set_value(InterpFunction &func, val,
                                   vector[DataType] &args) except -1
cdef int interp_function_eval_all(InterpFunction &func, unsigned age,
                                  py_object &pyage) except -1
cdef TagVal interp_function_call(InterpFunction &func) noexcept

cdef inline RuntimeValue _new_rtval(ValueType type_, DataType dt):
    # Avoid passing arguments to the constructor which requires the arguments
    # to be boxed and put in a tuple. This improves the performance by about 20%.
    self = <RuntimeValue>RuntimeValue.__new__(RuntimeValue)
    self.type_ = type_
    self.cache.type = dt
    self.age = -1
    return self

cdef RuntimeValue new_const(v)

cdef inline RuntimeValue new_arg(idx, ty=float):
    self = _new_rtval(ValueType.Arg, pytype_to_datatype(ty))
    self.cb_arg2 = idx
    return self

cpdef inline RuntimeValue new_extern(cb, ty=float):
    self = _new_rtval(ValueType.Extern, pytype_to_datatype(ty))
    self.cb_arg2 = cb
    return self

cpdef inline RuntimeValue new_extern_age(cb, ty=float):
    self = _new_rtval(ValueType.ExternAge, pytype_to_datatype(ty))
    self.cb_arg2 = cb
    return self

cdef RuntimeValue new_expr1(ValueType type_, RuntimeValue arg0)
cdef RuntimeValue new_expr2(ValueType type_, RuntimeValue arg0, RuntimeValue arg1)

cdef inline RuntimeValue round_int64_rt(RuntimeValue v):
    if v.type_ == ValueType.Int64:
        return v
    return new_expr1(ValueType.Int64, v)

cpdef convert_bool(v)
cpdef ifelse(b, v1, v2)

cdef inline bint is_rtval(v) noexcept:
    return type(v) is RuntimeValue

cdef _get_value(v, unsigned age, py_object &pyage)

cdef inline bint get_value_bool(v, unsigned age, py_object &pyage) except -1:
    if is_rtval(v):
        rt_eval_tagval(<RuntimeValue>v, age, pyage)
        return not (<RuntimeValue>v).cache.is_zero()
    else:
        return bool(v)

cdef inline double get_value_f64(v, unsigned age, py_object &pyage) except? -1:
    if is_rtval(v):
        rt_eval_tagval(<RuntimeValue>v, age, pyage)
        return (<RuntimeValue>v).cache.get[double]()
    else:
        return <double>v

cdef class ExternCallback:
    pass
