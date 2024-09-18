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

#ifndef BRASSBOARD_SEQ_SRC_RTVAL_H
#define BRASSBOARD_SEQ_SRC_RTVAL_H

#include "Python.h"

#include <stdint.h>

namespace brassboard_seq::rtval {

enum ValueType {
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
    if (type == (PyObject *)&PyInt_Type)
        return DataType::Int64;
    if (type == (PyObject *)&PyBool_Type)
        return DataType::Bool;
    PyErr_Format(PyExc_TypeError, "Unknown runtime value type '%S'", type);
    throw 0;
}

template<typename T> static constexpr DataType data_type_v = DataType::Bool;
template<> constexpr DataType data_type_v<bool> = DataType::Bool;
template<> constexpr DataType data_type_v<int64_t> = DataType::Int64;
template<> constexpr DataType data_type_v<double> = DataType::Float64;

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

    template<typename T> struct _getter;

    template<typename T>
    T &get()
    {
        return _getter<T>::get(*this);
    }
    template<typename T>
    const T &get() const
    {
        return _getter<T>::get(*this);
    }
};
template<> struct GenVal::_getter<bool> {
    static inline bool &get(GenVal &v) { return v.b_val; };
    static inline const bool &get(const GenVal &v) { return v.b_val; };
};
template<> struct GenVal::_getter<int64_t> {
    static inline int64_t &get(GenVal &v) { return v.i64_val; };
    static inline const int64_t &get(const GenVal &v) { return v.i64_val; };
};
template<> struct GenVal::_getter<double> {
    static inline double &get(GenVal &v) { return v.f64_val; };
    static inline const double &get(const GenVal &v) { return v.f64_val; };
};

struct TagVal {
    TagVal(bool b)
        : type(DataType::Bool),
          val{ .b_val = b }
    {}
    template<typename T>
    TagVal(T i, std::enable_if_t<std::is_integral_v<T>>* = nullptr)
        : type(DataType::Int64),
          val{ .i64_val = int64_t(i) }
    {}
    template<typename T>
    TagVal(T f, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr)
        : type(DataType::Float64),
          val{ .f64_val = double(f) }
    {}
    TagVal(DataType type=DataType::Bool)
        : type(type)
    {}
    DataType type;
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
    static TagVal from_py(PyObject *obj);
    PyObject *to_py() const;
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

}

#endif
