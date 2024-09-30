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

namespace brassboard_seq::rtval {

static inline TagVal tagval_add_or_sub(TagVal v1, TagVal v2, bool issub)
{
    assert(v1.err == EvalError::NoError);
    assert(v2.err == EvalError::NoError);
    return (issub ? Sub_op::generic_eval(v1, v2) : Add_op::generic_eval(v1, v2));
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_new_expr2_wrap1(PyObject *RTValueType, ValueType type,
                 PyObject *arg0, PyObject *arg1, RuntimeValue*)
{
    py_object rtarg0;
    py_object rtarg1;
    if (Py_TYPE(arg0) != (PyTypeObject*)RTValueType) {
        rtarg0.reset((PyObject*)new_const(RTValueType, arg0, (RuntimeValue*)nullptr));
        rtarg1.reset(py_newref(arg1));
    }
    else {
        if (Py_TYPE(arg1) == (PyTypeObject*)RTValueType) {
            rtarg1.reset(py_newref(arg1));
        }
        else {
            rtarg1.reset((PyObject*)new_const(RTValueType, arg1,
                                              (RuntimeValue*)nullptr));
        }
        rtarg0.reset(py_newref(arg0));
    }
    auto datatype = binary_return_type(type, ((RuntimeValue*)rtarg0.get())->datatype,
                                       ((RuntimeValue*)rtarg1.get())->datatype);
    auto o = throw_if_not(PyType_GenericAlloc((PyTypeObject*)RTValueType, 0));
    auto self = (RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)rtarg0.release();
    self->arg1 = (RuntimeValue*)rtarg1.release();
    self->cb_arg2 = py_newref(Py_None);
    return self;
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_wrap_rtval(PyObject *RTValueType, PyObject *v, RuntimeValue*)
{
    if (Py_TYPE(v) == (PyTypeObject*)RTValueType)
        return (RuntimeValue*)py_newref(v);
    return new_const(RTValueType, v, (RuntimeValue*)nullptr);
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_new_select(PyObject *RTValueType, RuntimeValue *arg0,
            PyObject *arg1, PyObject *arg2)
{
    py_object rtarg1((PyObject*)_wrap_rtval(RTValueType, arg1, (RuntimeValue*)nullptr));
    py_object rtarg2((PyObject*)_wrap_rtval(RTValueType, arg2, (RuntimeValue*)nullptr));
    auto datatype = promote_type(((RuntimeValue*)rtarg1.get())->datatype,
                                 ((RuntimeValue*)rtarg2.get())->datatype);
    auto o = throw_if_not(PyType_GenericAlloc((PyTypeObject*)RTValueType, 0));
    auto self = (RuntimeValue*)o;
    self->datatype = datatype;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = Select;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py_newref((PyObject*)arg0);
    self->arg1 = (RuntimeValue*)rtarg1.release();
    self->cb_arg2 = rtarg2.release();
    return self;
}

}
