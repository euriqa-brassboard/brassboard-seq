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

#include "utils.h"

namespace brassboard_seq::rtval {

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_new_cb_arg2(PyObject *RTValueType, ValueType type, PyObject *cb_arg2,
             PyObject *ty, RuntimeValue*)
{
    auto datatype = pytype_to_datatype(ty);
    auto o = throw_if_not(PyType_GenericAlloc((PyTypeObject*)RTValueType, 0));
    auto self = (RuntimeValue*)o;
    new (&self->cache) TagVal(datatype);
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py_newref(Py_None);
    self->arg1 = (RuntimeValue*)py_newref(Py_None);
    self->cb_arg2 = py_newref(cb_arg2);
    return self;
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_new_expr1(PyObject *RTValueType, ValueType type, RuntimeValue *arg0)
{
    auto o = throw_if_not(PyType_GenericAlloc((PyTypeObject*)RTValueType, 0));
    auto datatype = unary_return_type(type, arg0->cache.type);
    auto self = (RuntimeValue*)o;
    new (&self->cache) TagVal(datatype);
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py_newref((RuntimeValue*)arg0);
    self->arg1 = (RuntimeValue*)py_newref(Py_None);
    self->cb_arg2 = py_newref(Py_None);
    return self;
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
_new_expr2(PyObject *RTValueType, ValueType type, RuntimeValue *arg0,
           RuntimeValue *arg1)
{
    auto o = throw_if_not(PyType_GenericAlloc((PyTypeObject*)RTValueType, 0));
    auto datatype = binary_return_type(type, arg0->cache.type, arg1->cache.type);
    auto self = (RuntimeValue*)o;
    new (&self->cache) TagVal(datatype);
    self->type_ = type;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py_newref((PyObject*)arg0);
    self->arg1 = (RuntimeValue*)py_newref((PyObject*)arg1);
    self->cb_arg2 = py_newref(Py_None);
    return self;
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
new_const(PyObject *RTValueType, TagVal v, RuntimeValue*)
{
    auto o = throw_if_not(PyType_GenericAlloc((PyTypeObject*)RTValueType, 0));
    auto self = (RuntimeValue*)o;
    new (&self->cache) TagVal(v);
    self->type_ = Const;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py_newref(Py_None);
    self->arg1 = (RuntimeValue*)py_newref(Py_None);
    self->cb_arg2 = py_newref(Py_None);
    return self;
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
new_const(PyObject *RTValueType, PyObject *v, RuntimeValue*)
{
    return new_const(RTValueType, TagVal::from_py(v), (RuntimeValue*)nullptr);
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
rt_convert_bool(PyObject *RTValueType, RuntimeValue *v)
{
    if (v->type_ == Int64)
        v = v->arg0;
    if (v->cache.type == DataType::Bool)
        return (RuntimeValue*)py_newref((PyObject*)v);
    return _new_expr1(RTValueType, Bool, v);
}

template<typename RuntimeValue>
static inline __attribute__((returns_nonnull)) RuntimeValue*
rt_round_int64(PyObject *RTValueType, RuntimeValue *v)
{
    if (v->type_ == Int64)
        return (RuntimeValue*)py_newref((PyObject*)v);
    return _new_expr1(RTValueType, Int64, v);
}

}

#endif
