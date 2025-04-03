/*************************************************************************
 *   Copyright (c) 2024 - 2025 Yichao Yu <yyc1992@gmail.com>             *
 *                                                                       *
 *   This library is free software; you can redistribute it and/or       *
 *   modify it under the terms of the GNU Lesser General Public          *
 *   License as published by the Free Software Foundation; either        *
 *   version 3.0 of the License, or (at your option) any later version.  *
 *                                                                       *
 *   This library is distributed in the hope that it will be useful,     *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of      *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU    *
 *   Lesser General Public License for more details.                     *
 *                                                                       *
 *   You should have received a copy of the GNU Lesser General Public    *
 *   License along with this library. If not,                            *
 *   see <http://www.gnu.org/licenses/>.                                 *
 *************************************************************************/

#ifndef BRASSBOARD_SEQ_SRC_LIB_RTVAL_H
#define BRASSBOARD_SEQ_SRC_LIB_RTVAL_H

#include <concepts>

#include "utils.h"

namespace brassboard_seq::rtval {

enum ValueType: int8_t {
    Arg = -3,
    ExternAge = -2,
    Const = -1,
    Extern = 0,

    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    CmpLT = 5,
    CmpGT = 6,
    CmpLE = 7,
    CmpGE = 8,
    CmpNE = 9,
    CmpEQ = 10,
    And = 11,
    Or = 12,
    Xor = 13,
    Not = 14,
    Abs = 15,
    Ceil = 16,
    Exp = 17,
    Expm1 = 18,
    Floor = 19,
    Log = 20,
    Log1p = 21,
    Log2 = 22,
    Log10 = 23,
    Pow = 24,
    Sqrt = 25,
    Asin = 26,
    Acos = 27,
    Atan = 28,
    Atan2 = 29,
    Asinh = 30,
    Acosh = 31,
    Atanh = 32,
    Sin = 33,
    Cos = 34,
    Tan = 35,
    Sinh = 36,
    Cosh = 37,
    Tanh = 38,
    Hypot = 39,
    // Erf = 40,
    // Erfc = 41,
    // Gamma = 42,
    // Lgamma = 43,
    Rint = 44,
    Max = 45,
    Min = 46,
    Mod = 47,
    // Interp = 48,
    Select = 49,
    // Identity = 50,
    Int64 = 51,
    Bool = 52,
};

static inline ValueType pycmp2valcmp(int op)
{
    assert(op == Py_LT || op == Py_GT || op == Py_LE || op == Py_GE ||
           op == Py_EQ || op == Py_NE);
    switch (op) {
    default:
    case Py_LT: return CmpLT;
    case Py_GT: return CmpGT;
    case Py_LE: return CmpLE;
    case Py_GE: return CmpGE;
    case Py_NE: return CmpNE;
    case Py_EQ: return CmpEQ;
    }
}

enum class DataType: uint8_t {
    Bool,
    Int64,
    Float64,
};

static inline DataType pytype_to_datatype(PyObject *type)
{
    if (type == (PyObject *)&PyFloat_Type)
        return DataType::Float64;
    if (type == (PyObject *)&PyLong_Type)
        return DataType::Int64;
    if (type == (PyObject *)&PyBool_Type)
        return DataType::Bool;
    py_throw_format(PyExc_TypeError, "Unknown runtime value type '%S'", type);
}

template<typename T> static constexpr DataType data_type_v =
    (std::same_as<T,int64_t> ? DataType::Int64 :
     std::same_as<T,double> ? DataType::Float64 : DataType::Bool);

template<DataType DT> struct _data_type;
template<> struct _data_type<DataType::Bool> { using type = bool; };
template<> struct _data_type<DataType::Int64> { using type = int64_t; };
template<> struct _data_type<DataType::Float64> { using type = double; };
template<DataType DT> using data_type_t = typename _data_type<DT>::type;
static constexpr inline DataType promote_type(DataType t1, DataType t2)
{
    return DataType(std::max(uint8_t(t1), uint8_t(t2)));
}

union GenVal {
    bool b_val;
    int64_t i64_val;
    double f64_val;

    template<typename T>
    static auto &_get(auto &&g)
    {
        if constexpr (std::same_as<T,bool>) {
            return g.b_val;
        }
        else if constexpr (std::same_as<T,int64_t>) {
            return g.i64_val;
        }
        else if constexpr (std::same_as<T,double>) {
            return g.f64_val;
        }
    }

    template<typename T>
    T &get()
    {
        return _get<T>(*this);
    }
    template<typename T>
    const T &get() const
    {
        return _get<T>(*this);
    }
};

enum class EvalError: uint8_t {
    NoError = 0,
    ZeroDivide = 1,
    PowComplex = 2,
    LogicInexact = 3,
    LogNeg = 4,
    SqrtNeg = 5,
    TrigDomain = 6,
};

static inline void throw_py_error(EvalError err, uintptr_t key=uintptr_t(-1))
{
    switch (err) {
    case EvalError::ZeroDivide:
        bb_throw_format(PyExc_ZeroDivisionError, key, "division by zero");
    case EvalError::PowComplex:
        bb_throw_format(PyExc_ValueError, key, "power of negative number");
    case EvalError::LogicInexact:
        bb_throw_format(PyExc_ValueError, key,
                        "bitwise operation on floating point numbers");
    case EvalError::LogNeg:
        bb_throw_format(PyExc_ValueError, key, "log of negative number");
    case EvalError::SqrtNeg:
        bb_throw_format(PyExc_ValueError, key, "sqrt of negative number");
    case EvalError::TrigDomain:
        // Too lazy to think of a name...
        bb_throw_format(PyExc_ValueError, key, "math domain error");
    default:
    case EvalError::NoError:
        return;
    }
}

namespace {
template<typename E, typename ... Es>
struct ErrorCombiner {
    static inline E combine(E e, Es ... es)
    {
        if (e != EvalError::NoError) [[unlikely]]
            return e;
        return ErrorCombiner<Es...>::combine(es...);
    }
};
template<typename E>
struct ErrorCombiner<E> {
    static inline E combine(E e)
    {
        return e;
    }
};
}

template<typename ... Es>
static inline __attribute__((always_inline,flatten))
EvalError combine_error(Es ... es)
{
    return ErrorCombiner<Es...>::combine(es...);
}

struct TagVal {
    TagVal(bool b)
        : type(DataType::Bool),
          val{ .b_val = b }
    {}
    template<std::integral T>
    TagVal(T i)
        : type(DataType::Int64),
          val{ .i64_val = int64_t(i) }
    {}
    template<std::floating_point T>
    TagVal(T f)
        : type(DataType::Float64),
          val{ .f64_val = double(f) }
    {}
    TagVal(DataType type=DataType::Bool, EvalError err=EvalError::NoError)
        : type(type),
          err(err)
    {}
    DataType type;
    EvalError err{EvalError::NoError};
    GenVal val{ .i64_val = 0 };
    template<typename T> T get(void) const
    {
        switch (type) {
        case DataType::Bool:
            return T(val.get<bool>());
        case DataType::Int64:
            return T(val.get<int64_t>());
        case DataType::Float64:
            return T(val.get<double>());
        default:
            return T(false);
        }
    }
    TagVal convert(DataType new_type) const
    {
        if (new_type == type)
            return *this;
        if (err != EvalError::NoError)
            return { new_type, err };

        switch (new_type) {
        case DataType::Bool:
            return get<bool>();
        case DataType::Int64:
            return get<int64_t>();
        case DataType::Float64:
            return get<double>();
        default:
            return { new_type };
        }
    }
    static TagVal from_py(PyObject *obj);
    __attribute__((returns_nonnull))
    PyObject *to_py() const
    {
        throw_py_error(err);
        switch (type) {
        case DataType::Bool:
            return py_immref(val.b_val ? Py_True : Py_False);
        case DataType::Int64:
            return pylong_from_longlong(val.i64_val);
        default:
        case DataType::Float64:
            return pyfloat_from_double(val.f64_val);
        }
    }
    bool is_zero() const
    {
        switch (type) {
        case DataType::Bool:
            return !val.b_val;
        default:
            return val.i64_val == 0;
        }
    }

};

template<typename T>
struct bin_op {
    static inline __attribute__((flatten,always_inline)) TagVal
    generic_eval(TagVal tv1, TagVal tv2)
    {
#define HANDLE_BINARY(t1, t2)                                           \
        if (tv1.type == DataType::t1 && tv2.type == DataType::t2) {     \
            constexpr auto out_dt = T::return_type(DataType::t1, DataType::t2); \
            using T1 = data_type_t<DataType::t1>;                       \
            using T2 = data_type_t<DataType::t2>;                       \
            using Tout = data_type_t<out_dt>;                           \
            auto v1 = tv1.val.get<T1>();                                \
            auto v2 = tv2.val.get<T2>();                                \
            return T::template eval_err<Tout,T1,T2>(v1, v2);            \
        }
        HANDLE_BINARY(Bool, Bool);
        HANDLE_BINARY(Bool, Int64);
        HANDLE_BINARY(Bool, Float64);
        HANDLE_BINARY(Int64, Bool);
        HANDLE_BINARY(Int64, Int64);
        HANDLE_BINARY(Int64, Float64);
        HANDLE_BINARY(Float64, Bool);
        HANDLE_BINARY(Float64, Int64);
        HANDLE_BINARY(Float64, Float64);
#undef HANDLE_BINARY
        return {};
    }
};

template<typename T>
struct no_error_op {
    template<typename Tout, typename ...Ts>
    static inline TagVal eval_err(Ts ...vs)
    {
        Tout res = T::template eval<Tout,Ts...>(vs...);
        return TagVal(res);
    }
};

template<typename T>
struct promote_bin_op : bin_op<T> {
    static constexpr DataType return_type(DataType t1, DataType t2)
    {
        return promote_type(t1, t2);
    }
};

template<typename T>
struct bool_bin_op : bin_op<T> {
    static constexpr DataType return_type(DataType, DataType)
    {
        return DataType::Bool;
    }
};

template<typename T>
struct float_bin_op : bin_op<T> {
    static constexpr DataType return_type(DataType, DataType)
    {
        return DataType::Float64;
    }
};

template<typename T>
struct demote_int_bin_op : bin_op<T> {
    static constexpr DataType return_type(DataType t1, DataType t2)
    {
        auto t = promote_type(t1, t2);
        if (t == DataType::Float64)
            return DataType::Int64;
        return t;
    }
};

template<typename T>
struct promote_int_bin_op : bin_op<T> {
    static constexpr DataType return_type(DataType t1, DataType t2)
    {
        return promote_type(promote_type(t1, t2), DataType::Int64);
    }
};

struct Add_op : promote_int_bin_op<Add_op>, no_error_op<Add_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return v1 + v2;
    }
};

struct Sub_op : promote_int_bin_op<Sub_op>, no_error_op<Sub_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return v1 - v2;
    }
};

struct Mul_op : promote_int_bin_op<Mul_op>, no_error_op<Mul_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return Tout(v1) * Tout(v2);
    }
};

struct Div_op : float_bin_op<Div_op> {
    template<typename Tout, typename T1, typename T2>
    static inline TagVal eval_err(T1 v1, T2 v2)
    {
        if (v2 == 0)
            return { data_type_v<Tout>, EvalError::ZeroDivide };
        return TagVal(Tout(Tout(v1) / Tout(v2)));
    }
};

struct Mod_op : promote_int_bin_op<Mod_op> {
    template<typename Tout, typename T1, typename T2>
    static inline TagVal eval_err(T1 v1, T2 v2)
    {
        if (v2 == 0)
            return { data_type_v<Tout>, EvalError::ZeroDivide };
        Tout a = Tout(v1);
        Tout b = Tout(v2);
        Tout r;
        if constexpr (data_type_v<Tout> == DataType::Float64) {
            r = fmod(a, b);
            r += ((r != 0) & ((r < 0) ^ (b < 0))) * b;
        }
        else {
            r = a % b;
            r += ((r != 0) & ((r ^ b) < 0)) * b;
        }
        return TagVal(r);
    }
};

struct Min_op : promote_bin_op<Min_op>, no_error_op<Min_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return std::min(Tout(v1), Tout(v2));
    }
};

struct Max_op : promote_bin_op<Max_op>, no_error_op<Max_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return std::max(Tout(v1), Tout(v2));
    }
};

struct CmpLT_op : bool_bin_op<CmpLT_op>, no_error_op<CmpLT_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return v1 < v2;
    }
};

struct CmpGT_op : bool_bin_op<CmpGT_op>, no_error_op<CmpGT_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return v1 > v2;
    }
};

struct CmpLE_op : bool_bin_op<CmpLE_op>, no_error_op<CmpLE_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return v1 <= v2;
    }
};

struct CmpGE_op : bool_bin_op<CmpGE_op>, no_error_op<CmpGE_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return v1 >= v2;
    }
};

struct CmpEQ_op : bool_bin_op<CmpEQ_op>, no_error_op<CmpEQ_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return v1 == v2;
    }
};

struct CmpNE_op : bool_bin_op<CmpNE_op>, no_error_op<CmpNE_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return v1 != v2;
    }
};

struct Atan2_op : float_bin_op<Atan2_op>, no_error_op<Atan2_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return std::atan2(Tout(v1), Tout(v2));
    }
};

struct Hypot_op : float_bin_op<Hypot_op>, no_error_op<Hypot_op> {
    template<typename Tout, typename T1, typename T2>
    static inline Tout eval(T1 v1, T2 v2)
    {
        return std::hypot(Tout(v1), Tout(v2));
    }
};

struct Pow_op : float_bin_op<Pow_op> {
    template<typename Tout, typename T1, typename T2>
    static inline TagVal eval_err(T1 v1, T2 v2)
    {
        if constexpr (data_type_v<T2> == DataType::Bool)
            return TagVal(v2 ? Tout(v1) : Tout(1));
        Tout res = std::pow(Tout(v1), Tout(v2));
        if (!std::isfinite(res)) [[unlikely]] {
            if constexpr (data_type_v<T2> != DataType::Bool)
                if (v1 == 0 && v2 < 0)
                    return { data_type_v<Tout>, EvalError::ZeroDivide };
            if (!std::isnan(v1) && !std::isnan(v2)) {
                return { data_type_v<Tout>, EvalError::PowComplex };
            }
        }
        return TagVal(res);
    }
};

struct And_op : demote_int_bin_op<And_op> {
    template<typename Tout, typename T1, typename T2>
    static inline TagVal eval_err(T1 v1, T2 v2)
    {
        Tout _v1 = Tout(v1);
        Tout _v2 = Tout(v2);
        if (data_type_v<T1> == DataType::Float64 && _v1 != v1)
            return { data_type_v<Tout>, EvalError::LogicInexact };
        if (data_type_v<T2> == DataType::Float64 && _v2 != v2)
            return { data_type_v<Tout>, EvalError::LogicInexact };
        return TagVal(Tout(Tout(v1) & Tout(v2)));
    }
};

struct Or_op : demote_int_bin_op<Or_op> {
    template<typename Tout, typename T1, typename T2>
    static inline TagVal eval_err(T1 v1, T2 v2)
    {
        Tout _v1 = Tout(v1);
        Tout _v2 = Tout(v2);
        if (data_type_v<T1> == DataType::Float64 && _v1 != v1)
            return { data_type_v<Tout>, EvalError::LogicInexact };
        if (data_type_v<T2> == DataType::Float64 && _v2 != v2)
            return { data_type_v<Tout>, EvalError::LogicInexact };
        return TagVal(Tout(Tout(v1) | Tout(v2)));
    }
};

struct Xor_op : demote_int_bin_op<Xor_op> {
    template<typename Tout, typename T1, typename T2>
    static inline TagVal eval_err(T1 v1, T2 v2)
    {
        Tout _v1 = Tout(v1);
        Tout _v2 = Tout(v2);
        if (data_type_v<T1> == DataType::Float64 && _v1 != v1)
            return { data_type_v<Tout>, EvalError::LogicInexact };
        if (data_type_v<T2> == DataType::Float64 && _v2 != v2)
            return { data_type_v<Tout>, EvalError::LogicInexact };
        return TagVal(Tout(Tout(v1) ^ Tout(v2)));
    }
};

template<typename T>
struct uni_op {
    static inline __attribute__((flatten,always_inline)) TagVal
    generic_eval(TagVal tv)
    {
#define HANDLE_UNARY(t)                                                 \
        if (tv.type == DataType::t) {                                   \
            constexpr auto out_dt = T::return_type(DataType::t);        \
            using T1 = data_type_t<DataType::t>;                        \
            using Tout = data_type_t<out_dt>;                           \
            auto v = tv.val.get<T1>();                                  \
            return T::template eval_err<Tout,T1>(v);                    \
        }
        HANDLE_UNARY(Bool);
        HANDLE_UNARY(Int64);
        HANDLE_UNARY(Float64);
#undef HANDLE_UNARY
        return {};
    }
};

template<typename T>
struct bool_uni_op : uni_op<T> {
    static constexpr DataType return_type(DataType)
    {
        return DataType::Bool;
    }
};

template<typename T>
struct float_uni_op : uni_op<T> {
    static constexpr DataType return_type(DataType)
    {
        return DataType::Float64;
    }
};

struct Not_op : bool_uni_op<Not_op>, no_error_op<Not_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return not v1;
    }
};

struct Bool_op : bool_uni_op<Bool_op>, no_error_op<Bool_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return bool(v1);
    }
};

struct Abs_op : uni_op<Abs_op>, no_error_op<Abs_op> {
    static constexpr DataType return_type(DataType t1)
    {
        return t1;
    }
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        if constexpr (data_type_v<T1> == DataType::Bool) {
            // Clang warns about calling abs on the unsigned type bool ...
            return v1;
        }
        else {
            return std::abs(v1);
        }
    }
};

struct Int64_op : uni_op<Int64_op>, no_error_op<Int64_op> {
    static constexpr DataType return_type(DataType)
    {
        return DataType::Int64;
    }
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        if constexpr (data_type_v<T1> == DataType::Float64) {
            return Tout(std::llrint(v1));
        }
        else {
            return Tout(v1);
        }
    }
};

struct Ceil_op : float_uni_op<Ceil_op>, no_error_op<Ceil_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::ceil(Tout(v1));
    }
};

struct Rint_op : float_uni_op<Rint_op>, no_error_op<Rint_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::rint(Tout(v1));
    }
};

struct Floor_op : float_uni_op<Floor_op>, no_error_op<Floor_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::floor(Tout(v1));
    }
};

struct Exp_op : float_uni_op<Exp_op>, no_error_op<Exp_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::exp(Tout(v1));
    }
};

struct Expm1_op : float_uni_op<Expm1_op>, no_error_op<Expm1_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::expm1(Tout(v1));
    }
};

struct Log_op : float_uni_op<Log_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if (v1 <= 0)
            return { data_type_v<Tout>, EvalError::LogNeg };
        return TagVal(std::log(Tout(v1)));
    }
};

struct Log1p_op : float_uni_op<Log1p_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if constexpr (data_type_v<T1> != DataType::Bool)
            if (v1 <= -1)
                return { data_type_v<Tout>, EvalError::LogNeg };
        return TagVal(std::log1p(Tout(v1)));
    }
};

struct Log2_op : float_uni_op<Log2_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if (v1 <= 0)
            return { data_type_v<Tout>, EvalError::LogNeg };
        return TagVal(std::log2(Tout(v1)));
    }
};

struct Log10_op : float_uni_op<Log10_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if (v1 <= 0)
            return { data_type_v<Tout>, EvalError::LogNeg };
        return TagVal(std::log10(Tout(v1)));
    }
};

struct Sqrt_op : float_uni_op<Sqrt_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if constexpr (data_type_v<T1> != DataType::Bool)
            if (v1 < 0)
                return { data_type_v<Tout>, EvalError::SqrtNeg };
        return TagVal(std::sqrt(Tout(v1)));
    }
};

struct Asin_op : float_uni_op<Asin_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if constexpr (data_type_v<T1> != DataType::Bool)
            if (v1 < -1 || v1 > 1)
                return { data_type_v<Tout>, EvalError::TrigDomain };
        return TagVal(std::asin(Tout(v1)));
    }
};

struct Acos_op : float_uni_op<Acos_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if constexpr (data_type_v<T1> != DataType::Bool)
            if (v1 < -1 || v1 > 1)
                return { data_type_v<Tout>, EvalError::TrigDomain };
        return TagVal(std::acos(Tout(v1)));
    }
};

struct Atan_op : float_uni_op<Atan_op>, no_error_op<Atan_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::atan(Tout(v1));
    }
};

struct Asinh_op : float_uni_op<Asinh_op>, no_error_op<Asinh_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::asinh(Tout(v1));
    }
};

struct Acosh_op : float_uni_op<Acosh_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if (v1 < 1)
            return { data_type_v<Tout>, EvalError::TrigDomain };
        return TagVal(std::acosh(Tout(v1)));
    }
};

struct Atanh_op : float_uni_op<Atanh_op> {
    template<typename Tout, typename T1>
    static inline TagVal eval_err(T1 v1)
    {
        if constexpr (data_type_v<T1> != DataType::Bool)
            if (v1 <= -1 || v1 >= 1)
                return { data_type_v<Tout>, EvalError::TrigDomain };
        return TagVal(std::atanh(Tout(v1)));
    }
};

struct Sin_op : float_uni_op<Sin_op>, no_error_op<Sin_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::sin(Tout(v1));
    }
};

struct Cos_op : float_uni_op<Cos_op>, no_error_op<Cos_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::cos(Tout(v1));
    }
};

struct Tan_op : float_uni_op<Tan_op>, no_error_op<Tan_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::tan(Tout(v1));
    }
};

struct Sinh_op : float_uni_op<Sinh_op>, no_error_op<Sinh_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::sinh(Tout(v1));
    }
};

struct Cosh_op : float_uni_op<Cosh_op>, no_error_op<Cosh_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::cosh(Tout(v1));
    }
};

struct Tanh_op : float_uni_op<Tanh_op>, no_error_op<Tanh_op> {
    template<typename Tout, typename T1>
    static inline Tout eval(T1 v1)
    {
        return std::tanh(Tout(v1));
    }
};

static inline DataType unary_return_type(ValueType type, DataType t1)
{
    switch (type) {
#define HANDLE_UNARY(op) case op: return op##_op::return_type(t1)
        HANDLE_UNARY(Not);
        HANDLE_UNARY(Bool);
        HANDLE_UNARY(Abs);
        HANDLE_UNARY(Ceil);
        HANDLE_UNARY(Floor);
        HANDLE_UNARY(Exp);
        HANDLE_UNARY(Expm1);
        HANDLE_UNARY(Log);
        HANDLE_UNARY(Log1p);
        HANDLE_UNARY(Log2);
        HANDLE_UNARY(Log10);
        HANDLE_UNARY(Sqrt);
        HANDLE_UNARY(Asin);
        HANDLE_UNARY(Acos);
        HANDLE_UNARY(Atan);
        HANDLE_UNARY(Asinh);
        HANDLE_UNARY(Acosh);
        HANDLE_UNARY(Atanh);
        HANDLE_UNARY(Sin);
        HANDLE_UNARY(Cos);
        HANDLE_UNARY(Tan);
        HANDLE_UNARY(Sinh);
        HANDLE_UNARY(Cosh);
        HANDLE_UNARY(Tanh);
        HANDLE_UNARY(Rint);
        HANDLE_UNARY(Int64);
#undef HANDLE_UNARY
    default:
        return DataType::Float64;
    }
}

static inline DataType binary_return_type(ValueType type, DataType t1, DataType t2)
{
    switch (type) {
#define HANDLE_BINARY(op) case op: return op##_op::return_type(t1, t2)
        HANDLE_BINARY(Add);
        HANDLE_BINARY(Sub);
        HANDLE_BINARY(Mul);
        HANDLE_BINARY(Div);
        HANDLE_BINARY(Pow);
        HANDLE_BINARY(Mod);
        HANDLE_BINARY(And);
        HANDLE_BINARY(Or);
        HANDLE_BINARY(Xor);
        HANDLE_BINARY(CmpLT);
        HANDLE_BINARY(CmpGT);
        HANDLE_BINARY(CmpLE);
        HANDLE_BINARY(CmpGE);
        HANDLE_BINARY(CmpNE);
        HANDLE_BINARY(CmpEQ);
        HANDLE_BINARY(Hypot);
        HANDLE_BINARY(Atan2);
        HANDLE_BINARY(Max);
        HANDLE_BINARY(Min);
#undef HANDLE_BINARY
    default:
        return DataType::Float64;
    }
}

struct _RuntimeValue {
    PyObject_HEAD
    ValueType type_;
    DataType datatype;
    EvalError cache_err;
    unsigned int age;
    GenVal cache_val;
    _RuntimeValue *arg0;
    _RuntimeValue *arg1;
    PyObject *cb_arg2;
};
template<typename RuntimeValue>
static inline void assert_compatible_rtvalue()
{
    static_assert(sizeof(_RuntimeValue) == sizeof(RuntimeValue));
#define ASSERT_FIELD_OFFSET(name) \
    static_assert(offsetof(_RuntimeValue, name) == offsetof(RuntimeValue, name))
    ASSERT_FIELD_OFFSET(type_);
    ASSERT_FIELD_OFFSET(datatype);
    ASSERT_FIELD_OFFSET(cache_err);
    ASSERT_FIELD_OFFSET(age);
    ASSERT_FIELD_OFFSET(cache_val);
    ASSERT_FIELD_OFFSET(arg0);
    ASSERT_FIELD_OFFSET(arg1);
    ASSERT_FIELD_OFFSET(cb_arg2);
}

extern PyObject *RTVal_Type;

static inline bool is_rtval(PyObject *v)
{
    return Py_TYPE(v) == (PyTypeObject*)RTVal_Type;
}

__attribute__((returns_nonnull)) _RuntimeValue*
_new_cb_arg2(ValueType type, PyObject *cb_arg2, PyObject *ty);
template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
new_cb_arg2(ValueType type, PyObject *cb_arg2, PyObject *ty, RuntimeValue*)
{
    assert_compatible_rtvalue<RuntimeValue>();
    return (RuntimeValue*)_new_cb_arg2(type, cb_arg2, ty);
}

__attribute__((returns_nonnull)) _RuntimeValue*
_new_expr1(ValueType type, _RuntimeValue *arg0);
template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
new_expr1(ValueType type, RuntimeValue *arg0)
{
    assert_compatible_rtvalue<RuntimeValue>();
    return (RuntimeValue*)_new_expr1(type, (_RuntimeValue*)arg0);
}

__attribute__((returns_nonnull)) _RuntimeValue*
_new_expr2(ValueType type, _RuntimeValue *arg0, _RuntimeValue *arg1);
template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
new_expr2(ValueType type, RuntimeValue *arg0, RuntimeValue *arg1)
{
    assert_compatible_rtvalue<RuntimeValue>();
    return (RuntimeValue*)_new_expr2(type, (_RuntimeValue*)arg0, (_RuntimeValue*)arg1);
}

__attribute__((returns_nonnull)) _RuntimeValue *_new_const(TagVal v);
template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
new_const(TagVal v, RuntimeValue*)
{
    assert_compatible_rtvalue<RuntimeValue>();
    return (RuntimeValue*)_new_const(v);
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
new_const(PyObject *v, RuntimeValue*)
{
    assert_compatible_rtvalue<RuntimeValue>();
    return (RuntimeValue*)_new_const(TagVal::from_py(v));
}

__attribute__((returns_nonnull)) PyObject*
new_expr2_wrap1(ValueType type, PyObject *arg0, PyObject *arg1);

__attribute__((returns_nonnull)) _RuntimeValue*
_new_select(_RuntimeValue *arg0, PyObject *arg1, PyObject *arg2);
template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
new_select(RuntimeValue *arg0, PyObject *arg1, PyObject *arg2)
{
    assert_compatible_rtvalue<RuntimeValue>();
    return (RuntimeValue*)_new_select((_RuntimeValue*)arg0, arg1, arg2);
}

template<typename RuntimeValue>
static inline __attribute__((always_inline)) TagVal rtval_cache(RuntimeValue *rtval)
{
    TagVal cache;
    cache.type = rtval->datatype;
    cache.err = rtval->cache_err;
    cache.val = rtval->cache_val;
    return cache;
}

bool rt_same_value(PyObject *v1, PyObject *v2);

void _rt_eval_cache(_RuntimeValue *self, unsigned age, py_object &pyage);

template<typename RuntimeValue>
static inline __attribute__((always_inline))
void rt_eval_cache(RuntimeValue *self, unsigned age, py_object &pyage)
{
    assert_compatible_rtvalue<RuntimeValue>();
    _rt_eval_cache((_RuntimeValue*)self, age, pyage);
}

template<typename RuntimeValue>
static inline __attribute__((always_inline))
void rt_eval_throw(RuntimeValue *self, unsigned age, py_object &pyage,
                   uintptr_t key=uintptr_t(-1))
{
    try {
        rt_eval_cache(self, age, pyage);
    }
    catch (...) {
        if (key != uintptr_t(-1) && PyErr_Occurred())
            bb_reraise(key);
        throw;
    }
    throw_py_error(self->cache_err, key);
}

static inline double get_value_f64(PyObject *v, unsigned age, py_object &pyage)
{
    if (is_rtval(v)) {
        rt_eval_throw((_RuntimeValue*)v, age, pyage);
        return rtval_cache((_RuntimeValue*)v).get<double>();
    }
    return brassboard_seq::get_value_f64(v, -1);
}

std::pair<EvalError,GenVal> interpret_func(const int *code, GenVal *data,
                                           EvalError *errors);

struct InterpFunction {
    std::vector<int> code;
    std::vector<GenVal> data;
    std::vector<EvalError> errors;
    std::vector<void*> rt_vals;

    DataType ret_type;

    struct Builder {
        struct ValueInfo {
            bool is_const{false};
            bool dynamic{false};
            bool inited{false};
            int idx{-1};
            TagVal val;
        };
        int nargs;
        std::vector<DataType> &types;
        std::map<void*,ValueInfo> value_infos{};
    };

    int ensure_index(Builder::ValueInfo &info, Builder &builder)
    {
        if (info.idx >= 0)
            return info.idx;
        int idx = data.size();
        info.idx = idx;
        data.push_back(info.val.val);
        builder.types.push_back(info.val.type);
        return idx;
    }

    template<typename RuntimeValue>
    void set_value(RuntimeValue *value, std::vector<DataType> &args)
    {
        assert_compatible_rtvalue<RuntimeValue>();
        _set_value((_RuntimeValue*)value, args);
    }

    template<typename RuntimeValue>
    void set_value(RuntimeValue *value, std::vector<DataType> &&args)
    {
        set_value(value, args);
    }

    void _set_value(_RuntimeValue *value, std::vector<DataType> &args);
    Builder::ValueInfo &visit_value(_RuntimeValue *value, Builder &builder);

    void _eval_all(unsigned age, py_object &pyage);

    template<typename RuntimeValue>
    inline void eval_all(unsigned age, py_object &pyage, RuntimeValue*)
    {
        assert_compatible_rtvalue<RuntimeValue>();
        _eval_all(age, pyage);
    }

    TagVal call()
    {
        auto [err, val] = interpret_func(code.data(), data.data(), errors.data());
        TagVal res;
        res.type = ret_type;
        res.err = err;
        res.val = val;
        return res;
    }

};

}

#endif
