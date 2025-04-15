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

#include "yaml.h"

namespace brassboard_seq::scan {

static inline void check_non_empty_string_arg(PyObject *arg, const char *name)
{
    if (PyUnicode_CheckExact(arg) && PyUnicode_GET_LENGTH(arg))
        return;
    py_throw_format(PyExc_TypeError, "%s must be a string", name);
}

template<bool ovr>
static void merge_dict_into(py::dict tgt, py::dict src);

template<bool ovr>
static void set_dict(py::dict tgt, py::ptr<> key, py::ptr<> value)
{
    auto oldv = tgt.try_get(key);
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
            tgt.set(key, value);
        }
    }
    else {
        throw_pyerr();
        tgt.set(key, py::dict_deepcopy(value));
    }
}

template<bool ovr>
static void merge_dict_into(py::dict tgt, py::dict src)
{
    for (auto [key, val]: py::dict_iter(src)) {
        set_dict<ovr>(tgt, key, val);
    }
}

static inline py::dict_ref set_new_dict(py::dict dict, PyObject *fieldname)
{
    auto new_item = py::new_dict();
    dict.set(fieldname, new_item);
    return new_item;
}

inline py::ref<> ParamPack::ensure_visited()
{
    if (auto res = py::dict(visited).try_get(fieldname))
        return res.ref();
    return set_new_dict(visited, fieldname);
}

inline py::dict_ref ParamPack::ensure_dict()
{
    if (auto res = py::dict(values).try_get(fieldname)) {
        if (PyDict_Check(res))
            return res.ref();
        py_throw_format(PyExc_TypeError, "Cannot access value as parameter pack.");
    }
    return set_new_dict(values, fieldname);
}

inline __attribute__((returns_nonnull)) PyObject *ParamPack::get_value()
{
    auto res = py::dict(values).try_get(fieldname);
    if (!res)
        py_throw_format(PyExc_KeyError, "Value is not assigned");
    if (PyDict_Check(res))
        py_throw_format(PyExc_TypeError, "Cannot get parameter pack as value");
    py::dict(visited).set(fieldname, Py_True);
    return py::newref(res);
}

inline __attribute__((returns_nonnull)) PyObject*
ParamPack::get_value_default(PyObject *default_value)
{
    assert(!PyDict_Check(default_value));
    auto res = py::dict(values).try_get(fieldname);
    if (!res) {
        py::dict(values).set(fieldname, default_value);
        res = default_value;
    }
    else if (PyDict_Check(res)) {
        py_throw_format(PyExc_TypeError, "Cannot get parameter pack as value");
    }
    py::dict(visited).set(fieldname, Py_True);
    return py::newref(res);
}

static PyObject*
parampack_vectorcall(ParamPack *self, PyObject *const *args, size_t _nargs,
                     PyObject *_kwnames) try {
    // Supported syntax
    // () -> get value without default
    // (value) -> get value with default
    // (*dicts, **kwargs) -> get parameter pack with default
    auto nargs = PyVectorcall_NARGS(_nargs);
    auto kwnames = py::tuple(_kwnames);
    int nkws = kwnames ? kwnames.size() : 0;
    if (nkws == 0) {
        if (nargs == 0)
            return self->get_value();
        if (nargs == 1) {
            auto arg0 = args[0];
            if (!PyDict_Check(arg0)) {
                return self->get_value_default(arg0);
            }
        }
    }
    auto self_values = self->ensure_dict();
    for (int i = 0; i < nargs; i++) {
        auto arg = args[i];
        if (!PyDict_Check(arg))
            py_throw_format(PyExc_TypeError,
                            "Cannot use value as default value for parameter pack");
        merge_dict_into<false>(self_values, arg);
    }
    auto kwvalues = args + nargs;
    for (int i = 0; i < nkws; i++)
        set_dict<false>(self_values, kwnames.get(i), kwvalues[i]);
    return py::newref((PyObject*)self);
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

static inline py::ref<ParamPack> parampack_alloc()
{
    auto self = py::generic_alloc<ParamPack>();
    *(void**)(self.get() + 1) = (void*)parampack_vectorcall;
    return self;
}

static PyObject *parampack_new(PyObject*, PyObject *const *args, size_t _nargs,
                               PyObject *_kwnames) try {
    auto self = parampack_alloc();
    self->visited = py::new_dict().rel();
    self->fieldname = py::newref("root"_py);
    auto nargs = PyVectorcall_NARGS(_nargs);
    auto kwnames = py::tuple(_kwnames);
    int nkws = kwnames ? kwnames.size() : 0;
    self->values = py::new_dict().rel();
    if (!nargs && !nkws)
        return self.rel();
    auto kwargs = py::new_dict();
    py::dict(self->values).set("root"_py, kwargs);
    for (size_t i = 0; i < nargs; i++) {
        auto arg = args[i];
        if (!PyDict_Check(arg))
            py_throw_format(PyExc_TypeError,
                            "Cannot use value as default value for parameter pack");
        merge_dict_into<false>(kwargs, arg);
    }
    auto kwvalues = args + nargs;
    for (int i = 0; i < nkws; i++)
        set_dict<false>(kwargs, kwnames.get(i), kwvalues[i]);
    return self.rel();
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

static PyObject *parampack_str(PyObject *py_self)
{
    return cxx_catch([&] {
        auto self = (ParamPack*)py_self;
        auto fieldname = self->fieldname;
        auto values = py::dict(self->values);
        auto field = values.try_get(fieldname);
        if (!field)
            return py::newref("<Undefined>"_py);
        if (!PyDict_CheckExact(field))
            return PyObject_Str(field);
        return yaml::sprint(field);
    });
}

__attribute__((visibility("protected")))
ParamPack *ParamPack::new_empty()
{
    auto self = parampack_alloc();
    self->values = py::new_dict().rel();
    self->visited = py::new_dict().rel();
    self->fieldname = py::newref("root"_py);
    return self.rel();
}

static PySequenceMethods ParamPack_as_sequence = {
    .sq_contains = [] (PyObject *py_self, PyObject *key) -> int {
        auto self = (ParamPack*)py_self;
        auto fieldname = self->fieldname;
        auto values = py::dict(self->values);
        return cxx_catch([&] {
            auto field = values.try_get(fieldname);
            if (!field)
                return false;
            if (!PyDict_CheckExact(field))
                py_throw_format(PyExc_TypeError, "Scalar value does not have field");
            return py::dict(field).contains(key);
        });
    },
};

static inline bool is_slice_none(PyObject *key)
{
    if (!PySlice_Check(key))
        return false;
    auto slice = (PySliceObject*)key;
    return slice->start == Py_None && slice->stop == Py_None && slice->step == Py_None;
}

static PyMappingMethods ParamPack_as_mapping = {
    .mp_subscript = [] (PyObject *py_self, PyObject *key) -> PyObject* {
        if (!is_slice_none(key))
            return PyErr_Format(PyExc_ValueError, "Invalid index for ParamPack: %S", key);
        auto self = (ParamPack*)py_self;
        auto fieldname = self->fieldname;
        auto values = py::dict(self->values);
        return cxx_catch([&] {
            auto field = values.try_get(fieldname);
            if (!field)
                return py::new_dict();
            if (!PyDict_CheckExact(field))
                py_throw_format(PyExc_TypeError, "Cannot access value as parameter pack.");
            auto res = py::new_dict();
            for (auto [k, v]: py::dict_iter(field))
                res.set(k, py::dict_deepcopy(v));
            return res;
        });
    },
};

__attribute__((visibility("protected")))
PyTypeObject ParamPack::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.ParamPack",
    // extra space for the vectorcall pointer
    .tp_basicsize = sizeof(ParamPack) + sizeof(void*),
    .tp_dealloc = [] (PyObject *py_self) {
        PyObject_GC_UnTrack(py_self);
        Type.tp_clear(py_self);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    .tp_vectorcall_offset = sizeof(ParamPack),
    .tp_repr = parampack_str,
    .tp_as_sequence = &ParamPack_as_sequence,
    .tp_as_mapping = &ParamPack_as_mapping,
    .tp_call = PyVectorcall_Call,
    .tp_str = parampack_str,
    .tp_getattro = [] (PyObject *py_self, PyObject *name) {
        return cxx_catch([&] () -> PyObject* {
            check_non_empty_string_arg(name, "name");
            if (PyUnicode_READ_CHAR(name, 0) == '_')
                return PyObject_GenericGetAttr(py_self, name);
            auto self = (ParamPack*)py_self;
            auto res = parampack_alloc();
            res->values = self->ensure_dict().rel();
            res->visited = self->ensure_visited().rel();
            res->fieldname = py::newref(name);
            return res.rel();
        });
    },
    .tp_setattro = [] (PyObject *py_self, PyObject *name, PyObject *value) -> int {
        return cxx_catch([&] {
            check_non_empty_string_arg(name, "name");
            // To be consistent with __getattribute__
            if (PyUnicode_READ_CHAR(name, 0) == '_')
                py_throw_format(PyExc_AttributeError,
                                "'ParamPack' object has no attribute '%U'", name);
            if (!value)
                py_throw_format(PyExc_RuntimeError, "Deleting attribute not supported");
            auto self = (ParamPack*)py_self;
            auto self_values = self->ensure_dict();
            auto oldvalue = self_values.try_get(name);
            if (oldvalue) {
                auto was_dict = PyDict_CheckExact(oldvalue);
                auto is_dict = PyDict_Check(value);
                if (was_dict && !is_dict)
                    py_throw_format(PyExc_TypeError,
                                    "Cannot override parameter pack as value");
                if (!was_dict && is_dict)
                    py_throw_format(PyExc_TypeError,
                                    "Cannot override value as parameter pack");
                if (is_dict) {
                    merge_dict_into<true>(oldvalue, value);
                }
                else {
                    self_values.set(name, value);
                }
            }
            else {
                self_values.set(name, py::dict_deepcopy(value));
            }
            return 0;
        });
    },
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        auto self = (ParamPack*)py_self;
        Py_VISIT(self->values);
        return 0;
    },
    .tp_clear = [] (PyObject *py_self) {
        auto self = (ParamPack*)py_self;
        Py_CLEAR(self->values);
        Py_CLEAR(self->visited);
        Py_CLEAR(self->fieldname);
        return 0;
    },
    .tp_vectorcall = parampack_new,
};

static PyObject *get_visited(PyObject*, PyObject *const *args, Py_ssize_t nargs)
{
    return cxx_catch([&] {
        py_check_num_arg("get_visited", nargs, 1, 1);
        if (Py_TYPE(args[0]) != &ParamPack::Type)
            py_throw_format(PyExc_TypeError, "Wrong type for ParamPack");
        auto self = (ParamPack*)args[0];
        auto fieldname = self->fieldname;
        auto visited = py::dict(self->visited);
        if (auto res = visited.try_get(fieldname))
            return py::newref(res);
        if (auto value = py::dict(self->values).try_get(fieldname);
            value && PyDict_CheckExact(value))
            return set_new_dict(visited, fieldname).rel();
        Py_RETURN_FALSE;
    });
}
__attribute__((visibility("protected")))
PyMethodDef parampack_get_visited_method ={
    "get_visited", (PyCFunction)(void*)get_visited, METH_FASTCALL, 0};

// Helper function for functions that takes an optional parameter pack
static PyObject *get_param(PyObject*, PyObject *const *args, Py_ssize_t nargs)
{
    return cxx_catch([&] () -> PyObject* {
        py_check_num_arg("get_param", nargs, 1, 1);
        if (args[0] == Py_None)
            return ParamPack::new_empty();
        return py::newref(args[0]);
    });
}
__attribute__((visibility("protected")))
PyMethodDef parampack_get_param_method ={
    "get_param", (PyCFunction)(void*)get_param, METH_FASTCALL, 0};

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&ParamPack::Type) < 0);
}

}
