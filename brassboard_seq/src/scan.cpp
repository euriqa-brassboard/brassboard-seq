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

#include "scan.h"

namespace brassboard_seq::scan {

static void merge_dict_into(PyObject *tgt, PyObject *src, bool ovr)
{
    PyObject *key;
    PyObject *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(src, &pos, &key, &value)) {
        auto oldv = PyDict_GetItemWithError(tgt, key);
        if (oldv) {
            bool is_dict = PyDict_Check(value);
            bool was_dict = PyDict_Check(oldv);
            if (was_dict && !is_dict) {
                PyErr_Format(PyExc_TypeError,
                             "Cannot override parameter pack as value");
                throw 0;
            }
            else if (!was_dict && is_dict) {
                PyErr_Format(PyExc_TypeError,
                             "Cannot override value as parameter pack");
                throw 0;
            }
            else if (is_dict) {
                merge_dict_into(oldv, value, ovr);
            }
            else if (ovr) {
                throw_if_not(PyDict_SetItem(tgt, key, value) == 0);
            }
        }
        else {
            throw_if(PyErr_Occurred());
            py_object copied(pydict_deepcopy(value));
            throw_if_not(PyDict_SetItem(tgt, key, copied.get()) == 0);
        }
    }
}

static inline PyObject*
set_new_dict(PyObject *dict, PyObject *fieldname)
{
    py_object new_item(PyDict_New());
    if (!new_item || PyDict_SetItem(dict, fieldname, new_item.get()) < 0)
        return nullptr;
    return new_item.release();
}

template<typename ParamPack>
static PyObject *ensure_visited(ParamPack *self)
{
    auto fieldname = self->fieldname;
    auto self_visited = self->visited;
    auto visited = PyDict_GetItemWithError(self_visited, fieldname);
    if (visited)
        return py_newref(visited);
    if (PyErr_Occurred())
        return nullptr;
    return set_new_dict(self_visited, fieldname);
}

template<typename ParamPack>
static PyObject *ensure_dict(ParamPack *self)
{
    auto fieldname = self->fieldname;
    auto self_values = self->values;
    auto values = PyDict_GetItemWithError(self_values, fieldname);
    if (values) {
        if (PyDict_Check(values))
            return py_newref(values);
        return PyErr_Format(PyExc_TypeError,
                            "Cannot access value as parameter pack.");
    }
    if (PyErr_Occurred())
        return nullptr;
    return set_new_dict(self_values, fieldname);
}

// Return borrowed reference
template<typename ParamPack>
static PyObject *_ensure_dict_kws(ParamPack *self, PyObject *kws)
{
    auto fieldname = self->fieldname;
    auto self_values = self->values;
    auto values = PyDict_GetItemWithError(self_values, fieldname);
    if (values) {
        if (PyDict_Check(values))
            return values;
        PyErr_Format(PyExc_TypeError,
                     "Cannot access value as parameter pack.");
        throw 0;
    }
    throw_if(PyErr_Occurred());
    throw_if_not(PyDict_SetItem(self_values, fieldname, kws) == 0);
    return kws;
}

template<typename ParamPack>
static PyObject *get_value(ParamPack *self)
{
    auto fieldname = self->fieldname;
    auto self_values = self->values;
    auto values = PyDict_GetItemWithError(self_values, fieldname);
    if (!values)
        return PyErr_Format(PyExc_KeyError, "Value is not assigned");
    if (PyDict_Check(values))
        return PyErr_Format(PyExc_TypeError, "Cannot get parameter pack as value");
    if (PyDict_SetItem(self->visited, fieldname, Py_True) < 0)
        return nullptr;
    return py_newref(values);
}

template<typename ParamPack>
static PyObject *get_value_default(ParamPack *self, PyObject *default_value)
{
    assert(!PyDict_Check(default_value));
    auto fieldname = self->fieldname;
    auto self_values = self->values;
    auto values = PyDict_GetItemWithError(self_values, fieldname);
    if (!values) {
        if (PyErr_Occurred() ||
            PyDict_SetItem(self_values, fieldname, default_value) < 0)
            return nullptr;
        values = default_value;
    }
    else if (PyDict_Check(values)) {
        return PyErr_Format(PyExc_TypeError, "Cannot get parameter pack as value");
    }
    if (PyDict_SetItem(self->visited, fieldname, Py_True) < 0)
        return nullptr;
    return py_newref(values);
}

template<typename ParamPack>
static PyObject *parampack_call(ParamPack *self, PyObject *args, PyObject *kwargs)
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
                return PyErr_Format(
                    PyExc_TypeError,
                    "Cannot use value as default value for parameter pack");
            merge_dict_into(self_values, arg, true);
        }
    }
    else {
        for (int i = 0; i < nargs; i++) {
            auto arg = PyTuple_GET_ITEM(args, i);
            if (!PyDict_Check(arg))
                return PyErr_Format(
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
