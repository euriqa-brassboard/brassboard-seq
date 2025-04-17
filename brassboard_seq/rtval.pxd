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
    # Cython doesn't seem to allow namespace in the object property
    # for the imported extension class
    """
    using _brassboard_seq_rtval_RuntimeValue = brassboard_seq::rtval::RuntimeValue;
    using _brassboard_seq_rtval_ExternCallback = brassboard_seq::rtval::ExternCallback;
    """
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

    enum class DataType(uint8_t):
        Bool
        Int64
        Float64

    union GenVal:
        bint b_val
        int64_t i64_val
        double f64_val

    enum class EvalError(uint8_t):
        pass
    void throw_py_error(EvalError) except +

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

    cppclass rtval_ref:
        RuntimeValue rel "rel<brassboard_seq::rtval::RuntimeValue>" ()

    RuntimeValue new_arg(object idx, object ty) except +
    RuntimeValue new_extern(ExternCallback cb, ty) except +
    RuntimeValue new_extern_age(ExternCallback cb, ty) except +
    rtval_ref new_expr2(ValueType, RuntimeValue, RuntimeValue) except +
    rtval_ref new_const(object) except +

    void rt_eval_throw(RuntimeValue self, unsigned age) except +
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

    ctypedef class brassboard_seq._utils.ExternCallback [object _brassboard_seq_rtval_ExternCallback, check_size ignore]:
        cdef void *fptr
