/*************************************************************************
 *   Copyright (c) 2024 - 2024 Yichao Yu <yyc1992@gmail.com>             *
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

#include "rtval.h"

#include "utils.h"

#include <cmath>

#include "numpy/arrayobject.h"

namespace brassboard_seq::rtval {

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
        if (unlikely(std::isnan(res))) {
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
            return Tout(v1 >= 0 ? v1 + 0.5 : v1 - 0.5);
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
        if constexpr (data_type_v<T1> != DataType::Bool)
            if (v1 > -1 && v1 < 1)
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

struct Select_op {
    static constexpr DataType return_type(DataType, DataType t2, DataType t3)
    {
        return promote_type(t2, t3);
    }
    static inline __attribute__((flatten,always_inline)) TagVal
    generic_eval(TagVal tv0, TagVal tv1, TagVal tv2)
    {
        bool b = bool(tv0.val.i64_val);
#define HANDLE_BINARY(t1, t2)                                           \
        if (tv1.type == DataType::t1 && tv2.type == DataType::t2) {     \
            constexpr auto out_dt = promote_type(DataType::t1, DataType::t2); \
            using T1 = data_type_t<DataType::t1>;                       \
            using T2 = data_type_t<DataType::t2>;                       \
            using Tout = data_type_t<out_dt>;                           \
            auto v1 = Tout(tv1.val.get<T1>());                          \
            auto v2 = Tout(tv2.val.get<T2>());                          \
            auto _res = b ? v1 : v2;                                    \
            TagVal res{};                                               \
            res.type = out_dt;                                          \
            res.val.get<Tout>() = _res;                                 \
            return res;                                                 \
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

template<typename RuntimeValue>
static inline __attribute__((flatten))
void rt_eval_cache(RuntimeValue *self, unsigned age, py_object &pyage)
{
    if (self->age == age)
        return;

    // Take the reference from the argument
    auto set_cache = [&] (TagVal v) {
        assert(v.type == self->cache.type);
        assume(v.type == self->cache.type);
        self->cache = v;
        self->age = age;
    };
    auto set_cache_py = [&] (PyObject *obj) {
        throw_if_not(obj);
        set_cache(TagVal::from_py(obj).convert(self->cache.type));
        Py_DECREF(obj);
    };

    auto type = self->type_;
    switch (type) {
    case Const:
        return;
    case Extern:
        set_cache_py(_PyObject_Vectorcall(self->cb_arg2, nullptr, 0, nullptr));
        return;
    case ExternAge: {
        if (!pyage)
            pyage.reset(throw_if_not(PyLong_FromLong(age)));
        PyObject *args[] = { pyage.get() };
        set_cache_py(_PyObject_Vectorcall(self->cb_arg2, args, 1, nullptr));
        return;
    }
    default:
        break;
    }

    auto rtarg0 = self->arg0;
    rt_eval_cache(rtarg0, age, pyage);
    auto arg0 = rtarg0->cache;
    auto eval1 = [&] (auto op_cls) {
        if (arg0.err != EvalError::NoError) {
            set_cache({ self->cache.type, arg0.err });
        }
        else {
            set_cache(op_cls.generic_eval(arg0));
        }
    };

    switch (type) {
#define HANDLE_UNARY(op) case op: eval1(op##_op()); return
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
        break;
    }

    auto rtarg1 = self->arg1;
    if (type == Select) {
        auto rtarg2 = (RuntimeValue*)self->cb_arg2;
        auto rtres = arg0.template get<bool>() ? rtarg1 : rtarg2;
        rt_eval_cache(rtres, age, pyage);
        set_cache(rtres->cache.convert(self->cache.type));
        return;
    }
    rt_eval_cache(rtarg1, age, pyage);
    auto arg1 = rtarg1->cache;

    auto eval2 = [&] (auto op_cls) {
        if (auto err = combine_error(arg0.err, arg1.err); err != EvalError::NoError) {
            set_cache({ self->cache.type, err });
        }
        else {
            set_cache(op_cls.generic_eval(arg0, arg1));
        }
    };

    switch (type) {
#define HANDLE_BINARY(op) case op: eval2(op##_op()); return
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
        PyErr_Format(PyExc_ValueError, "Unknown value type");
        throw 0;
    }
}

static inline bool is_numpy_int(PyObject *value)
{
    if (PyArray_IsScalar(value, Integer))
        return true;
    return PyArray_IsZeroDim(value) && PyArray_ISINTEGER((PyArrayObject*)value);
}

inline TagVal TagVal::from_py(PyObject *value)
{
    if (value == Py_True)
        return true;
    if (value == Py_False)
        return false;
    if (PyLong_Check(value) || is_numpy_int(value)) {
        auto val = PyLong_AsLong(value);
        throw_if(val == -1 && PyErr_Occurred());
        return TagVal(val);
    }
    auto val = PyFloat_AsDouble(value);
    throw_if(val == -1 && PyErr_Occurred());
    return TagVal(val);
}

inline PyObject *TagVal::to_py() const
{
    throw_py_error(err);
    switch (type) {
    case DataType::Bool:
        return py_newref(val.b_val ? Py_True : Py_False);
    case DataType::Int64:
        return PyLong_FromLong(val.i64_val);
    default:
    case DataType::Float64:
        return pyfloat_from_double(val.f64_val);
    }
}

static inline TagVal tagval_add_or_sub(TagVal v1, TagVal v2, bool issub)
{
    assert(v1.err == EvalError::NoError);
    assert(v2.err == EvalError::NoError);
    return (issub ? Sub_op::generic_eval(v1, v2) : Add_op::generic_eval(v1, v2));
}

}
