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

static void merge_dict_into(PyObject *tgt, PyObject *src, bool ovr)
{
    foreach_pydict(src, [&] (auto key, auto value) {
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
                merge_dict_into(oldv, value, ovr);
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
    });
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

// Return borrowed reference
static __attribute__((returns_nonnull)) PyObject*
_ensure_dict_kws(auto *self, PyObject *kws)
{
    auto fieldname = self->fieldname;
    auto self_values = self->values;
    auto values = PyDict_GetItemWithError(self_values, fieldname);
    if (values) {
        if (PyDict_Check(values))
            return values;
        py_throw_format(PyExc_TypeError, "Cannot access value as parameter pack.");
    }
    throw_if(PyErr_Occurred());
    throw_if(PyDict_SetItem(self_values, fieldname, kws));
    return kws;
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

static __attribute__((returns_nonnull)) PyObject*
parampack_call(auto *self, PyObject *args, PyObject *kwargs)
{
    int nargs = PyTuple_GET_SIZE(args);
    int nkws = PyDict_GET_SIZE(kwargs);
    if (nkws == 0) {
        if (nargs == 0)
            return get_value(self);
        if (nargs == 1) {
            auto arg0 = PyTuple_GET_ITEM(args, 0);
            if (!PyDict_Check(arg0)) {
                return get_value_default(self, arg0);
            }
        }
    }
    // Reuse the kwargs dict if possible
    auto self_values = _ensure_dict_kws(self, kwargs);
    if (self_values == kwargs) {
        for (int i = 0; i < nargs; i++) {
            auto arg = PyTuple_GET_ITEM(args, nargs - 1 - i);
            if (!PyDict_Check(arg))
                py_throw_format(
                    PyExc_TypeError,
                    "Cannot use value as default value for parameter pack");
            merge_dict_into(self_values, arg, true);
        }
    }
    else {
        for (int i = 0; i < nargs; i++) {
            auto arg = PyTuple_GET_ITEM(args, i);
            if (!PyDict_Check(arg))
                py_throw_format(
                    PyExc_TypeError,
                    "Cannot use value as default value for parameter pack");
            merge_dict_into(self_values, arg, false);
        }
        if (nkws) {
            merge_dict_into(self_values, kwargs, false);
        }
    }
    return py_newref((PyObject*)self);
}

}
