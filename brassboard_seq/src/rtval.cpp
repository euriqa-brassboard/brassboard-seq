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

static inline __attribute__((returns_nonnull)) PyObject*
_new_addsub(TagVal c, _RuntimeValue *v, bool s)
{
    if (c.is_zero() && !s)
        return py_newref((PyObject*)v);
    py_object arg0((PyObject*)_new_const(c));
    return (PyObject*)_new_expr2(s ? Sub : Add, (_RuntimeValue*)arg0.get(), v);
}

static inline __attribute__((returns_nonnull)) PyObject*
build_addsub(PyObject *v0, PyObject *v1, bool issub)
{
    assume(v0);
    assume(v1);
    TagVal nc{};
    _RuntimeValue *nv0 = nullptr;
    bool ns0 = false;
    if (!is_rtval(v0)) {
        nc = TagVal::from_py(v0);
        assert(is_rtval(v1));
        assume(is_rtval(v1));
    }
    else {
        nv0 = (_RuntimeValue*)v0;
        auto type = nv0->type_;
        if (type == Const) {
            nc = rtval_cache(nv0);
            nv0 = nullptr;
        }
        else if (type == Add) {
            auto arg0 = nv0->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                nc = rtval_cache(arg0);
                nv0 = nv0->arg1;
            }
        }
        else if (type == Sub) {
            auto arg0 = nv0->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                ns0 = true;
                nc = rtval_cache(arg0);
                nv0 = nv0->arg1;
            }
        }
    }
    bool ns1 = false;
    _RuntimeValue *nv1 = nullptr;
    if (!is_rtval(v1)) {
        nc = tagval_add_or_sub(nc, TagVal::from_py(v1), issub);
    }
    else {
        nv1 = (_RuntimeValue*)v1;
        auto type = nv1->type_;
        if (type == Const) {
            nc = tagval_add_or_sub(nc, rtval_cache(nv1), issub);
            nv1 = nullptr;
        }
        else if (type == Add) {
            auto arg0 = nv1->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                nc = tagval_add_or_sub(nc, rtval_cache(arg0), issub);
                nv1 = nv1->arg1;
            }
        }
        else if (type == Sub) {
            auto arg0 = nv1->arg0;
            // Add/Sub should only have the first argument as constant
            if (arg0->type_ == Const) {
                ns1 = true;
                nc = tagval_add_or_sub(nc, rtval_cache(arg0), issub);
                nv1 = nv1->arg1;
            }
        }
    }
    if ((PyObject*)nv0 == v0 && (PyObject*)nv1 == v1)
        return (PyObject*)_new_expr2(issub ? Sub : Add, nv0, nv1);
    if (issub)
        ns1 = !ns1;
    if (!nv0) {
        if (!nv1)
            return (PyObject*)_new_const(nc);
        return _new_addsub(nc, nv1, ns1);
    }
    if (!nv1)
        return _new_addsub(nc, nv0, ns0);
    bool ns = false;
    py_object nv;
    if (!ns0) {
        nv.reset((PyObject*)_new_expr2(ns1 ? Sub : Add, nv0, nv1));
    }
    else if (ns1) {
        nv.reset((PyObject*)_new_expr2(Add, nv0, nv1));
        ns = true;
    }
    else {
        nv.reset((PyObject*)_new_expr2(Sub, nv1, nv0));
    }
    return _new_addsub(nc, (_RuntimeValue*)nv.get(), ns);
}

}
