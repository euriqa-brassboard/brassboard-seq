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

    TagVal rtval_cache(RuntimeValue)

    bint is_rtval(object)
    bint rt_same_value(object, object)

    RuntimeValue new_cb_arg2(ValueType, object, object ty, RuntimeValue) except +
    RuntimeValue new_expr1(ValueType, RuntimeValue) except +
    RuntimeValue new_expr2(ValueType, RuntimeValue, RuntimeValue) except +
    RuntimeValue new_const(object, RuntimeValue) except +
    RuntimeValue new_select(RuntimeValue arg0, object, object) except +
    RuntimeValue rt_convert_bool(RuntimeValue) except +
    RuntimeValue rt_round_int64(RuntimeValue) except +

    void rt_eval_cache(RuntimeValue self, unsigned age) except +
    void rt_eval_throw(RuntimeValue self, unsigned age) except +
    void rt_eval_throw(RuntimeValue self, unsigned age, uintptr_t) except +
    double get_value_f64(object, unsigned age) except +

    cppclass InterpFunction:
        void set_value(RuntimeValue, vector[DataType]&) except +
        void eval_all(unsigned) except +
        TagVal call()

    ctypedef class brassboard_seq._utils.RuntimeValue [object _brassboard_seq_rtval_RuntimeValue, check_size ignore]:
        cdef ValueType type_
        cdef DataType datatype
        cdef EvalError cache_err
        cdef unsigned age
        cdef GenVal cache_val
        cdef RuntimeValue arg0
        cdef RuntimeValue arg1
        cdef object cb_arg2 # Also used as argument index

cdef extern from *:
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_rtval_RuntimeValue = brassboard_seq::rtval::_RuntimeValue;
    """

cdef inline RuntimeValue new_arg(idx, ty):
    return new_cb_arg2(ValueType.Arg, idx, ty, None)

cdef inline RuntimeValue new_extern(ExternCallback cb, ty):
    return new_cb_arg2(ValueType.Extern, cb, ty, None)

cdef inline RuntimeValue new_extern_age(ExternCallback cb, ty):
    return new_cb_arg2(ValueType.ExternAge, cb, ty, None)

cdef class ExternCallback:
    cdef void *fptr
