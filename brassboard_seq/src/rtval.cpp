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

static PyObject *__pyx_f_14brassboard_seq_5rtval_call0(PyObject *f);
static PyObject *__pyx_f_14brassboard_seq_5rtval_call1(PyObject *f, PyObject *arg0);
static PyObject *__pyx_f_14brassboard_seq_5rtval_call2(PyObject *f, PyObject *arg0,
                                                       PyObject *arg1);
static PyObject *__pyx_f_14brassboard_seq_5rtval__round_int64(PyObject *f);

namespace brassboard_seq::rtval {

static PyObject *cnpy_ceil;
static PyObject *cnpy_exp;
static PyObject *cnpy_expm1;
static PyObject *cnpy_floor;
static PyObject *cnpy_log;
static PyObject *cnpy_log1p;
static PyObject *cnpy_log2;
static PyObject *cnpy_log10;
static PyObject *cnpy_sqrt;
static PyObject *cnpy_arcsin;
static PyObject *cnpy_arccos;
static PyObject *cnpy_arctan;
static PyObject *cnpy_arcsinh;
static PyObject *cnpy_arccosh;
static PyObject *cnpy_arctanh;
static PyObject *cnpy_sin;
static PyObject *cnpy_cos;
static PyObject *cnpy_tan;
static PyObject *cnpy_sinh;
static PyObject *cnpy_cosh;
static PyObject *cnpy_tanh;
static PyObject *cnpy_rint;
static PyObject *cnpy_hypot;
static PyObject *cnpy_arctan2;

template<typename RuntimeValue>
static __attribute__((flatten))
void _rt_eval_cache(RuntimeValue *self, unsigned age, py_object<PyObject> &pyage)
{
    if (self->age == age)
        return;

    auto get_pyage = [&] {
        if (!pyage)
            pyage.reset(throw_if_not(PyLong_FromLong(age)));
        return pyage.get();
    };

    // Take the reference from the argument
    auto set_cache = [&] (PyObject *v) {
        throw_if_not(v);
        auto oldv = self->cache;
        self->cache = v;
        Py_DECREF(oldv);
        self->age = age;
    };

    auto type = self->type_;
    switch (type) {
    case Const:
        return;
    case Extern:
        set_cache(__pyx_f_14brassboard_seq_5rtval_call0(self->cb_arg2));
        return;
    case ExternAge:
        set_cache(__pyx_f_14brassboard_seq_5rtval_call1(self->cb_arg2, get_pyage()));
        return;
    default:
        break;
    }

    auto rtarg0 = (RuntimeValue*)self->arg0;
    _rt_eval_cache(rtarg0, age, pyage);
    auto eval1 = [&] (auto &&cb) {
        set_cache(cb(rtarg0->cache));
    };
    auto eval_call1 = [&] (PyObject *func) {
        set_cache(__pyx_f_14brassboard_seq_5rtval_call1(func, rtarg0->cache));
    };

    switch (type) {
    case Not:
        eval1([&] (auto arg0) {
            return py_newref(get_value_bool(arg0, uintptr_t(-1)) ? Py_False : Py_True);
        });
        return;
    case Bool:
        eval1([&] (auto arg0) {
            return py_newref(get_value_bool(arg0, uintptr_t(-1)) ? Py_True : Py_False);
        });
        return;
    case Abs:
        eval1(PyNumber_Absolute);
        return;
    case Ceil:
        eval_call1(cnpy_ceil);
        return;
    case Floor:
        eval_call1(cnpy_floor);
        return;
    case Exp:
        eval_call1(cnpy_exp);
        return;
    case Expm1:
        eval_call1(cnpy_expm1);
        return;
    case Log:
        eval_call1(cnpy_log);
        return;
    case Log1p:
        eval_call1(cnpy_log1p);
        return;
    case Log2:
        eval_call1(cnpy_log2);
        return;
    case Log10:
        eval_call1(cnpy_log10);
        return;
    case Sqrt:
        eval_call1(cnpy_sqrt);
        return;
    case Asin:
        eval_call1(cnpy_arcsin);
        return;
    case Acos:
        eval_call1(cnpy_arccos);
        return;
    case Atan:
        eval_call1(cnpy_arctan);
        return;
    case Asinh:
        eval_call1(cnpy_arcsinh);
        return;
    case Acosh:
        eval_call1(cnpy_arccosh);
        return;
    case Atanh:
        eval_call1(cnpy_arctanh);
        return;
    case Sin:
        eval_call1(cnpy_sin);
        return;
    case Cos:
        eval_call1(cnpy_cos);
        return;
    case Tan:
        eval_call1(cnpy_tan);
        return;
    case Sinh:
        eval_call1(cnpy_sinh);
        return;
    case Cosh:
        eval_call1(cnpy_cosh);
        return;
    case Tanh:
        eval_call1(cnpy_tanh);
        return;
    case Rint:
        eval_call1(cnpy_rint);
        return;
    case Int64:
        eval1(__pyx_f_14brassboard_seq_5rtval__round_int64);
        return;
    default:
        break;
    }

    auto rtarg1 = (RuntimeValue*)self->arg1;
    if (type == Select) {
        auto rtarg2 = (RuntimeValue*)self->cb_arg2;
        auto rtres = get_value_bool(rtarg0->cache, uintptr_t(-1)) ? rtarg1 : rtarg2;
        _rt_eval_cache(rtres, age, pyage);
        set_cache(py_newref(rtres->cache));
        return;
    }
    _rt_eval_cache(rtarg1, age, pyage);

    auto arg0 = rtarg0->cache;
    auto arg1 = rtarg1->cache;
    auto eval_call2 = [&] (PyObject *func) {
        set_cache(__pyx_f_14brassboard_seq_5rtval_call2(func, arg0, arg1));
    };

    switch (type) {
    case Add:
        set_cache(PyNumber_Add(arg0, arg1));
        return;
    case Sub:
        set_cache(PyNumber_Subtract(arg0, arg1));
        return;
    case Mul:
        set_cache(PyNumber_Multiply(arg0, arg1));
        return;
    case Div:
        set_cache(PyNumber_TrueDivide(arg0, arg1));
        return;
    case Pow:
        set_cache(PyNumber_Power(arg0, arg1, Py_None));
        return;
    case Mod:
        set_cache(PyNumber_Remainder(arg0, arg1));
        return;
    case And:
        set_cache(PyNumber_And(arg0, arg1));
        return;
    case Or:
        set_cache(PyNumber_Or(arg0, arg1));
        return;
    case Xor:
        set_cache(PyNumber_Xor(arg0, arg1));
        return;
    case CmpLT:
        set_cache(PyObject_RichCompare(arg0, arg1, Py_LT));
        return;
    case CmpGT:
        set_cache(PyObject_RichCompare(arg0, arg1, Py_GT));
        return;
    case CmpLE:
        set_cache(PyObject_RichCompare(arg0, arg1, Py_LE));
        return;
    case CmpGE:
        set_cache(PyObject_RichCompare(arg0, arg1, Py_GE));
        return;
    case CmpNE:
        set_cache(PyObject_RichCompare(arg0, arg1, Py_NE));
        return;
    case CmpEQ:
        set_cache(PyObject_RichCompare(arg0, arg1, Py_EQ));
        return;
    case Hypot:
        eval_call2(cnpy_hypot);
        return;
    case Atan2:
        eval_call2(cnpy_arctan2);
        return;
    case Max:
    case Min: {
        py_object cmp(throw_if_not(PyObject_RichCompare(arg0, arg1, Py_LT)));
        auto res = ((get_value_bool(cmp.get(), uintptr_t(-1)) xor (type == Max)) ?
                    arg0 : arg1);
        set_cache(py_newref(res));
        return;
    }
    default:
        PyErr_Format(PyExc_ValueError, "Unknown value type");
        throw 0;
    }
}

template<typename RuntimeValue>
static inline  __attribute__((always_inline))
void rt_eval_cache(RuntimeValue *self, unsigned age)
{
    py_object<PyObject> pyage;
    _rt_eval_cache(self, age, pyage);
}

}
