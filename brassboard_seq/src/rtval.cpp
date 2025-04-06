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

static TagVal rtprop_callback_func(auto *self, unsigned age)
{
    py_object v(throw_if_not(PyObject_GetAttr(self->obj, self->fieldname)));
    if (!is_rtval(v))
        return TagVal::from_py(v);
    auto rv = (_RuntimeValue*)v.get();
    if (rv->type_ == ExternAge && rv->cb_arg2 == (PyObject*)self)
        py_throw_format(PyExc_ValueError, "RT property have not been assigned.");
    rt_eval_cache(rv, age);
    return rtval_cache(rv);
}

template<typename composite_rtprop_data>
static inline __attribute__((returns_nonnull)) composite_rtprop_data*
get_composite_rtprop_data(auto prop, PyObject *obj, PyObject *DataType,
                          composite_rtprop_data*)
{
    auto fieldname = prop->fieldname;
    if (fieldname == Py_None)
        py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
    if (py_object val(PyObject_GetAttr(obj, fieldname)); !val) {
        PyErr_Clear();
    }
    else if (Py_TYPE(val.get()) == (PyTypeObject*)DataType) {
        return (composite_rtprop_data*)val.release();
    }
    py_object o(pytype_genericalloc(DataType));
    auto data = (composite_rtprop_data*)o.get();
    data->ovr = py_immref(Py_None);
    data->cache = py_immref(Py_None);
    throw_if(PyObject_SetAttr(obj, fieldname, o.get()));
    return (composite_rtprop_data*)o.release();
}

static inline __attribute__((returns_nonnull)) PyObject*
apply_composite_ovr(PyObject *val, PyObject *ovr);

static inline bool
apply_dict_ovr(PyObject *dict, PyObject *k, PyObject *v)
{
    auto field = PyDict_GetItemWithError(dict, k);
    if (field) {
        py_object newfield(apply_composite_ovr(field, v));
        throw_if(PyDict_SetItem(dict, k, newfield));
        return true;
    }
    throw_if(PyErr_Occurred());
    return false;
}

static inline __attribute__((returns_nonnull)) PyObject*
apply_composite_ovr(PyObject *val, PyObject *ovr)
{
    if (!PyDict_Check(ovr))
        return py_newref(ovr);
    if (!PyDict_Size(ovr))
        return py_newref(val);
    if (PyDict_Check(val)) {
        py_object newval(throw_if_not(PyDict_Copy(val)));
        for (auto [k, v]: pydict_iter(ovr)) {
            if (apply_dict_ovr(newval, k, v))
                continue;
            // for scangroup support since only string key is supported
            if (py_object ik(PyNumber_Long(k)); !ik) {
                PyErr_Clear();
            }
            else if (apply_dict_ovr(newval, ik, v)) {
                continue;
            }
            throw_if(PyDict_SetItem(newval, k, v));
        }
        return newval.release();
    }
    if (PyList_Check(val)) {
        py_object newval(throw_if_not(PySequence_List(val)));
        for (auto [k, v]: pydict_iter(ovr)) {
            // for scangroup support since only string key is supported
            py_object ik(throw_if_not(PyNumber_Long(k)));
            auto idx = PyLong_AsLong(ik.get());
            if (idx < 0) {
                throw_if(PyErr_Occurred());
                py_throw_format(PyExc_IndexError, "list index out of range");
            }
            if (idx >= PyList_GET_SIZE(newval.get()))
                py_throw_format(PyExc_IndexError, "list index out of range");
            auto olditem = PyList_GET_ITEM(newval.get(), idx);
            PyList_SET_ITEM(newval.get(), idx, apply_composite_ovr(olditem, v));
            Py_DECREF(olditem);
        }
        return newval.release();
    }
    py_throw_format(PyExc_TypeError, "Unknown value type '%S'", Py_TYPE(val));
}

static PyObject *_bb_rt_values_str = pyunicode_from_string("_bb_rt_values");

static inline bool _object_compiled(PyObject *obj)
{
    py_object field(PyObject_GetAttr(obj, _bb_rt_values_str));
    if (!field)
        PyErr_Clear();
    return field.get() == Py_None;
}

template<typename composite_rtprop_data>
static inline __attribute__((returns_nonnull)) PyObject*
composite_rtprop_get_res(auto self, PyObject *obj, PyObject *DataType,
                         composite_rtprop_data*)
{
    auto data = get_composite_rtprop_data<composite_rtprop_data>(self, obj,
                                                                 DataType, nullptr);
    py_object py_data((PyObject*)data);
    if (!data->filled || (!data->compiled && _object_compiled(obj))) {
        py_object res(throw_if_not(_PyObject_Vectorcall(self->cb, &obj, 1, nullptr)));
        Py_DECREF(data->cache);
        data->cache = res.release();
        data->filled = true;
        data->compiled = _object_compiled(obj);
    }
    if (data->ovr == Py_None)
        return py_newref(data->cache);
    return apply_composite_ovr(data->cache, data->ovr);
}

}
