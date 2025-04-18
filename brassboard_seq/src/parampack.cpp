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
    if (auto s = py::cast<py::str>(arg); s && s.size())
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
        bool is_dict = value.isa<py::dict>();
        bool was_dict = oldv.isa<py::dict>();
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
        if (res.isa<py::dict>())
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
    if (res.isa<py::dict>())
        py_throw_format(PyExc_TypeError, "Cannot get parameter pack as value");
    py::dict(visited).set(fieldname, Py_True);
    return py::newref(res);
}

inline __attribute__((returns_nonnull)) PyObject*
ParamPack::get_value_default(PyObject *default_value)
{
    assert(!py::isa<py::dict>(default_value));
    auto res = py::dict(values).try_get(fieldname);
    if (!res) {
        py::dict(values).set(fieldname, default_value);
        res = default_value;
    }
    else if (res.isa<py::dict>()) {
        py_throw_format(PyExc_TypeError, "Cannot get parameter pack as value");
    }
    py::dict(visited).set(fieldname, Py_True);
    return py::newref(res);
}

static PyObject *parampack_vectorcall(py::ptr<ParamPack> self, PyObject *const *args,
                                      ssize_t nargs, py::tuple kwnames)
{
    // Supported syntax
    // () -> get value without default
    // (value) -> get value with default
    // (*dicts, **kwargs) -> get parameter pack with default
    int nkws = kwnames ? kwnames.size() : 0;
    if (nkws == 0) {
        if (nargs == 0)
            return self->get_value();
        if (nargs == 1) {
            auto arg0 = args[0];
            if (!py::isa<py::dict>(arg0)) {
                return self->get_value_default(arg0);
            }
        }
    }
    auto self_values = self->ensure_dict();
    for (int i = 0; i < nargs; i++) {
        auto arg = args[i];
        if (!py::isa<py::dict>(arg))
            py_throw_format(PyExc_TypeError,
                            "Cannot use value as default value for parameter pack");
        merge_dict_into<false>(self_values, arg);
    }
    auto kwvalues = args + nargs;
    for (int i = 0; i < nkws; i++)
        set_dict<false>(self_values, kwnames.get(i), kwvalues[i]);
    return py::newref((PyObject*)self);
}

static inline py::ref<ParamPack> parampack_alloc()
{
    auto self = py::generic_alloc<ParamPack>();
    *(void**)(self.get() + 1) = (void*)py::vectorfunc<parampack_vectorcall>;
    return self;
}

static auto parampack_new(PyObject*, PyObject *const *args, ssize_t nargs,
                          py::tuple kwnames)
{
    auto self = parampack_alloc();
    self->visited = py::new_dict().rel();
    self->fieldname = py::newref("root"_py);
    int nkws = kwnames ? kwnames.size() : 0;
    self->values = py::new_dict().rel();
    if (!nargs && !nkws)
        return self;
    auto kwargs = py::new_dict();
    py::dict(self->values).set("root"_py, kwargs);
    for (size_t i = 0; i < nargs; i++) {
        auto arg = args[i];
        if (!py::isa<py::dict>(arg))
            py_throw_format(PyExc_TypeError,
                            "Cannot use value as default value for parameter pack");
        merge_dict_into<false>(kwargs, arg);
    }
    auto kwvalues = args + nargs;
    for (int i = 0; i < nkws; i++)
        set_dict<false>(kwargs, kwnames.get(i), kwvalues[i]);
    return self;
}

static constexpr auto parampack_str = py::unifunc<[] (py::ptr<ParamPack> self) {
    auto fieldname = self->fieldname;
    auto values = py::dict(self->values);
    auto field = values.try_get(fieldname);
    if (!field)
        return py::newref("<Undefined>"_py);
    if (!field.typeis<py::dict>())
        return PyObject_Str(field);
    return yaml::sprint(field);
}>;

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
    .sq_contains = py::ibinfunc<[] (py::ptr<ParamPack> self, PyObject *key) -> int {
        auto fieldname = self->fieldname;
        auto values = py::dict(self->values);
        auto field = values.try_get(fieldname);
        if (!field)
            return false;
        if (!field.typeis<py::dict>())
            py_throw_format(PyExc_TypeError, "Scalar value does not have field");
        return py::dict(field).contains(key);
    }>,
};

static inline bool is_slice_none(PyObject *key)
{
    if (!PySlice_Check(key))
        return false;
    auto slice = (PySliceObject*)key;
    return slice->start == Py_None && slice->stop == Py_None && slice->step == Py_None;
}

static PyMappingMethods ParamPack_as_mapping = {
    .mp_subscript = py::binfunc<[] (py::ptr<ParamPack> self, py::ptr<> key) {
        if (!is_slice_none(key))
            py_throw_format(PyExc_ValueError, "Invalid index for ParamPack: %S", key);
        auto fieldname = self->fieldname;
        auto values = py::dict(self->values);
        auto field = values.try_get(fieldname);
        if (!field)
            return py::new_dict();
        if (!field.typeis<py::dict>())
            py_throw_format(PyExc_TypeError, "Cannot access value as parameter pack.");
        auto res = py::new_dict();
        for (auto [k, v]: py::dict_iter(field))
            res.set(k, py::dict_deepcopy(v));
        return res;
    }>,
};

__attribute__((visibility("protected")))
PyTypeObject ParamPack::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.ParamPack",
    // extra space for the vectorcall pointer
    .tp_basicsize = sizeof(ParamPack) + sizeof(void*),
    .tp_dealloc = py::tp_dealloc<true,[] (PyObject *self) { Type.tp_clear(self); }>,
    .tp_vectorcall_offset = sizeof(ParamPack),
    .tp_repr = parampack_str,
    .tp_as_sequence = &ParamPack_as_sequence,
    .tp_as_mapping = &ParamPack_as_mapping,
    .tp_call = PyVectorcall_Call,
    .tp_str = parampack_str,
    .tp_getattro = py::binfunc<[] (py::ptr<ParamPack> self, py::ptr<> name) -> py::ref<> {
        check_non_empty_string_arg(name, "name");
        if (PyUnicode_READ_CHAR(name, 0) == '_')
            return py::ref(PyObject_GenericGetAttr(self, name));
        auto res = parampack_alloc();
        res->values = self->ensure_dict().rel();
        res->visited = self->ensure_visited().rel();
        res->fieldname = py::newref(name);
        return res;
    }>,
    .tp_setattro = py::itrifunc<[] (py::ptr<ParamPack> self, py::ptr<> name, py::ptr<> value) {
        check_non_empty_string_arg(name, "name");
        // To be consistent with __getattribute__
        if (PyUnicode_READ_CHAR(name, 0) == '_')
            py_throw_format(PyExc_AttributeError,
                            "'ParamPack' object has no attribute '%U'", name);
        if (!value)
            py_throw_format(PyExc_RuntimeError, "Deleting attribute not supported");
        auto self_values = self->ensure_dict();
        auto oldvalue = self_values.try_get(name);
        if (oldvalue) {
            auto was_dict = oldvalue.typeis<py::dict>();
            auto is_dict = py::isa<py::dict>(value);
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
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC|Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_traverse = py::tp_traverse<[] (py::ptr<ParamPack> self, auto &visitor) {
        visitor(self->values);
    }>,
    .tp_clear = py::iunifunc<[] (py::ptr<ParamPack> self) {
        py::CLEAR(self->values);
        py::CLEAR(self->visited);
        py::CLEAR(self->fieldname);
    }>,
    .tp_vectorcall = py::vectorfunc<parampack_new>,
};

static inline py::ref<> get_visited(PyObject*, py::ptr<> param_pack)
{
    auto self = py::arg_cast<ParamPack,true>(param_pack, "param_pack");
    auto fieldname = self->fieldname;
    auto visited = py::dict(self->visited);
    if (auto res = visited.try_get(fieldname))
        return res.ref();
    if (auto value = py::dict(self->values).try_get(fieldname);
        value && value.typeis<py::dict>())
        return set_new_dict(visited, fieldname);
    return py::new_false();
}
__attribute__((visibility("protected")))
PyMethodDef parampack_get_visited_method = py::meth_o<"get_visited",get_visited>;

// Helper function for functions that takes an optional parameter pack
static inline PyObject *get_param(PyObject*, py::ptr<> param)
{
    if (param == Py_None)
        return ParamPack::new_empty();
    return py::newref(param);
}
__attribute__((visibility("protected")))
PyMethodDef parampack_get_param_method = py::meth_o<"get_param",get_param>;

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&ParamPack::Type) < 0);
}

}
