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

#include "scan.h"

namespace brassboard_seq::scan {

template<bool ovr>
static void merge_dict_into(PyObject *tgt, PyObject *src);

template<bool ovr>
static void set_dict(PyObject *tgt, PyObject *key, PyObject *value)
{
    auto oldv = PyDict_GetItemWithError(tgt, key);
    if (oldv) {
        bool is_dict = PyDict_Check(value);
        bool was_dict = PyDict_Check(oldv);
        if (was_dict && !is_dict) {
            py_throw_format(PyExc_TypeError,
                            "Cannot override parameter pack as value");
        }
        else if (!was_dict && is_dict) {
            py_throw_format(PyExc_TypeError,
                            "Cannot override value as parameter pack");
        }
        else if (is_dict) {
            merge_dict_into<ovr>(oldv, value);
        }
        else if (ovr) {
            throw_if(PyDict_SetItem(tgt, key, value));
        }
    }
    else {
        throw_if(PyErr_Occurred());
        py_object copied(pydict_deepcopy(value));
        throw_if(PyDict_SetItem(tgt, key, copied.get()));
    }
}

template<bool ovr>
static void merge_dict_into(PyObject *tgt, PyObject *src)
{
    for (auto [key, val]: pydict_iter(src)) {
        set_dict<ovr>(tgt, key, val);
    }
}

static inline void merge_dict_ovr(PyObject *tgt, PyObject *src)
{
    merge_dict_into<true>(tgt, src);
}

static inline __attribute__((returns_nonnull)) PyObject*
set_new_dict(PyObject *dict, PyObject *fieldname)
{
    py_object new_item(pydict_new());
    throw_if(PyDict_SetItem(dict, fieldname, new_item.get()));
    return new_item.release();
}

static __attribute__((returns_nonnull)) PyObject *ensure_visited(auto *self)
{
    auto fieldname = self->fieldname;
    auto self_visited = self->visited;
    auto visited = PyDict_GetItemWithError(self_visited, fieldname);
    if (visited)
        return py_newref(visited);
    throw_if(PyErr_Occurred());
    return set_new_dict(self_visited, fieldname);
}

static __attribute__((returns_nonnull)) PyObject *ensure_dict(auto *self)
{
    auto fieldname = self->fieldname;
    auto self_values = self->values;
    auto values = PyDict_GetItemWithError(self_values, fieldname);
    if (values) {
        if (PyDict_Check(values))
            return py_newref(values);
        py_throw_format(PyExc_TypeError, "Cannot access value as parameter pack.");
    }
    throw_if(PyErr_Occurred());
    return set_new_dict(self_values, fieldname);
}

static __attribute__((returns_nonnull)) PyObject *get_value(auto *self)
{
    auto fieldname = self->fieldname;
    auto self_values = self->values;
    auto values = PyDict_GetItemWithError(self_values, fieldname);
    if (!values)
        return PyErr_Format(PyExc_KeyError, "Value is not assigned");
    if (PyDict_Check(values))
        return PyErr_Format(PyExc_TypeError, "Cannot get parameter pack as value");
    throw_if(PyDict_SetItem(self->visited, fieldname, Py_True));
    return py_newref(values);
}

static __attribute__((returns_nonnull)) PyObject*
get_value_default(auto *self, PyObject *default_value)
{
    assert(!PyDict_Check(default_value));
    auto fieldname = self->fieldname;
    auto self_values = self->values;
    auto values = PyDict_GetItemWithError(self_values, fieldname);
    if (!values) {
        throw_if(PyErr_Occurred());
        throw_if(PyDict_SetItem(self_values, fieldname, default_value));
        values = default_value;
    }
    else if (PyDict_Check(values)) {
        py_throw_format(PyExc_TypeError, "Cannot get parameter pack as value");
    }
    throw_if(PyDict_SetItem(self->visited, fieldname, Py_True));
    return py_newref(values);
}

// Check if the struct field reference path is overwritten in `obj`.
// Overwrite happens if the field itself exists or a parent of the field
// is overwritten to something that's not scalar struct.
static inline bool check_field(PyObject *d, PyObject *path)
{
    for (auto [_, name]: pytuple_iter(path)) {
        auto vp = PyDict_GetItemWithError(d, name);
        if (!vp) {
            throw_if(PyErr_Occurred());
            return false;
        }
        if (!PyDict_CheckExact(vp))
            return true;
        d = vp;
    }
    return true;
}

static PyObject*
parampack_vectorcall(auto *self, PyObject *const *args, size_t _nargs,
                     PyObject *kwnames) try {
    auto nargs = PyVectorcall_NARGS(_nargs);
    int nkws = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    if (nkws == 0) {
        if (nargs == 0)
            return get_value(self);
        if (nargs == 1) {
            auto arg0 = args[0];
            if (!PyDict_Check(arg0)) {
                return get_value_default(self, arg0);
            }
        }
    }
    auto self_values = ensure_dict(self);
    for (int i = 0; i < nargs; i++) {
        auto arg = args[i];
        if (!PyDict_Check(arg))
            py_throw_format(
                PyExc_TypeError,
                "Cannot use value as default value for parameter pack");
        merge_dict_into<false>(self_values, arg);
    }
    auto kwvalues = args + nargs;
    for (int i = 0; i < nkws; i++)
        set_dict<false>(self_values, PyTuple_GET_ITEM(kwnames, i), kwvalues[i]);
    return py_newref((PyObject*)self);
}
catch (...) {
    return nullptr;
}

template<typename ParamPack> static PyObject*
parampack_new(PyObject *type, PyObject *const *args, size_t _nargs,
              PyObject *kwnames) try {
    py_object o(pytype_genericalloc(type));
    auto self = (ParamPack*)o.get();
    self->vectorcall_ptr = (void*)&parampack_vectorcall<ParamPack>;
    self->visited = pydict_new();
    self->fieldname = py_newref("root"_py);
    auto nargs = PyVectorcall_NARGS(_nargs);
    int nkws = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    self->values = pydict_new();
    if (!nargs && !nkws)
        return o.release();
    py_object kwargs(pydict_new());
    throw_if(PyDict_SetItem(self->values, "root"_py, kwargs.get()));
    for (size_t i = 0; i < nargs; i++) {
        auto arg = args[i];
        if (!PyDict_Check(arg))
            py_throw_format(PyExc_TypeError,
                            "Cannot use value as default value for parameter pack");
        merge_dict_into<false>(kwargs, arg);
    }
    auto kwvalues = args + nargs;
    for (int i = 0; i < nkws; i++)
        set_dict<false>(kwargs, PyTuple_GET_ITEM(kwnames, i), kwvalues[i]);
    return o.release();
}
catch (...) {
    return nullptr;
}

template<typename ParamPack>
static inline void update_param_pack(PyObject *_type, ParamPack*)
{
    auto type = (PyTypeObject*)_type;
    type->tp_vectorcall_offset = offsetof(ParamPack, vectorcall_ptr);
    type->tp_flags |= Py_TPFLAGS_HAVE_VECTORCALL;
    type->tp_call = PyVectorcall_Call;
    type->tp_alloc = [] (PyTypeObject *t, Py_ssize_t nitems) -> PyObject* {
        auto o = (ParamPack*)PyType_GenericAlloc(t, nitems);
        if (o) [[likely]]
            o->vectorcall_ptr = (void*)&parampack_vectorcall<ParamPack>;
        return (PyObject*)o;
    };
    type->tp_vectorcall = parampack_new<ParamPack>;
    PyType_Modified(type);
}

}
