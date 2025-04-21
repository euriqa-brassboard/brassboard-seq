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

static inline void check_non_empty_string_arg(py::ptr<> arg, const char *name)
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

static inline py::dict_ref set_new_dict(py::dict dict, py::str fieldname)
{
    auto new_item = py::new_dict();
    dict.set(fieldname, new_item);
    return new_item;
}

inline py::ref<> ParamPack::ensure_visited()
{
    if (auto res = visited.try_get(fieldname))
        return res.ref();
    return set_new_dict(visited, fieldname);
}

inline py::dict_ref ParamPack::ensure_dict()
{
    if (auto res = values.try_get(fieldname)) {
        if (res.isa<py::dict>())
            return res.ref();
        py_throw_format(PyExc_TypeError, "Cannot access value as parameter pack.");
    }
    return set_new_dict(values, fieldname);
}

static py::ref<> parampack_vectorcall(py::ptr<ParamPack> self, PyObject *const *args,
                                      ssize_t nargs, py::tuple kwnames)
{
    // Supported syntax
    // () -> get value without default
    // (value) -> get value with default
    // (*dicts, **kwargs) -> get parameter pack with default
    int nkws = kwnames ? kwnames.size() : 0;
    if (nkws == 0) {
        if (nargs == 0) {
            auto res = self->values.try_get(self->fieldname);
            if (!res)
                py_throw_format(PyExc_KeyError, "Value is not assigned");
            if (res.isa<py::dict>())
                py_throw_format(PyExc_TypeError, "Cannot get parameter pack as value");
            self->visited.set(self->fieldname, Py_True);
            return res.ref();
        }
        if (nargs == 1) {
            auto arg0 = py::ptr(args[0]);
            if (!py::isa<py::dict>(arg0)) {
                auto res = self->values.try_get(self->fieldname);
                if (!res) {
                    self->values.set(self->fieldname, arg0);
                    res = arg0;
                }
                else if (res.isa<py::dict>()) {
                    py_throw_format(PyExc_TypeError, "Cannot get parameter pack as value");
                }
                self->visited.set(self->fieldname, Py_True);
                return res.ref();
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
    return self.ref();
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
    auto self = ParamPack::new_empty();
    int nkws = kwnames ? kwnames.size() : 0;
    if (!nargs && !nkws)
        return self;
    auto kwargs = py::new_dict();
    self->values.set("root"_py, kwargs);
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
    auto field = self->values.try_get(self->fieldname);
    if (!field)
        return py::newref("<Undefined>"_py);
    if (!field.typeis<py::dict>())
        return PyObject_Str(field);
    return yaml::sprint(field);
}>;

__attribute__((visibility("protected")))
py::ref<ParamPack> ParamPack::new_empty()
{
    auto self = parampack_alloc();
    call_constructor(&self->values, py::new_dict());
    call_constructor(&self->visited, py::new_dict());
    call_constructor(&self->fieldname, "root"_py.ref());
    return self;
}

__attribute__((visibility("protected")))
PyTypeObject ParamPack::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.ParamPack",
    // extra space for the vectorcall pointer
    .tp_basicsize = sizeof(ParamPack) + sizeof(void*),
    .tp_dealloc = py::tp_dealloc<true,[] (py::ptr<ParamPack> self) {
        call_destructor(&self->values);
        call_destructor(&self->visited);
        call_destructor(&self->fieldname);
    }>,
    .tp_vectorcall_offset = sizeof(ParamPack),
    .tp_repr = parampack_str,
    .tp_as_sequence = &global_var<PySequenceMethods{
        .sq_contains = py::ibinfunc<[] (py::ptr<ParamPack> self, py::ptr<> key) {
            auto field = self->values.try_get(self->fieldname);
            if (!field)
                return false;
            if (!field.typeis<py::dict>())
                py_throw_format(PyExc_TypeError, "Scalar value does not have field");
            return py::dict(field).contains(key);
        }>,
    }>,
    .tp_as_mapping = &global_var<PyMappingMethods{
        .mp_subscript = py::binfunc<[] (py::ptr<ParamPack> self, py::ptr<> key) {
            if (!py::is_slice_none(key))
                py_throw_format(PyExc_ValueError, "Invalid index for ParamPack: %S", key);
            auto field = self->values.try_get(self->fieldname);
            if (!field)
                return py::new_dict();
            if (!field.typeis<py::dict>())
                py_throw_format(PyExc_TypeError, "Cannot access value as parameter pack.");
            auto res = py::new_dict();
            for (auto [k, v]: py::dict_iter(field))
                res.set(k, py::dict_deepcopy(v));
            return res;
        }>,
    }>,
    .tp_call = PyVectorcall_Call,
    .tp_str = parampack_str,
    .tp_getattro = py::binfunc<[] (py::ptr<ParamPack> self, py::ptr<> name) -> py::ref<> {
        check_non_empty_string_arg(name, "name");
        if (PyUnicode_READ_CHAR(name, 0) == '_')
            return py::ref(PyObject_GenericGetAttr(self, name));
        auto res = parampack_alloc();
        call_constructor(&res->values, self->ensure_dict());
        call_constructor(&res->visited, self->ensure_visited());
        call_constructor(&res->fieldname, name.ref());
        return res;
    }>,
    .tp_setattro = py::itrifunc<[] (py::ptr<ParamPack> self, py::ptr<> name,
                                    py::ptr<> value) {
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
        self->values.CLEAR();
        self->visited.CLEAR();
        self->fieldname.CLEAR();
    }>,
    .tp_vectorcall = py::vectorfunc<parampack_new>,
};

static inline py::ref<> get_visited(PyObject*, py::ptr<> param_pack)
{
    auto self = py::arg_cast<ParamPack,true>(param_pack, "param_pack");
    auto fieldname = self->fieldname.ptr();
    if (auto res = self->visited.try_get(fieldname))
        return res.ref();
    if (auto value = self->values.try_get(fieldname);
        value && value.typeis<py::dict>())
        return set_new_dict(self->visited, fieldname);
    return py::new_false();
}
__attribute__((visibility("protected")))
PyMethodDef parampack_get_visited_method = py::meth_o<"get_visited",get_visited>;

// Helper function for functions that takes an optional parameter pack
static inline py::ref<> get_param(PyObject*, py::ptr<> param)
{
    if (param == Py_None)
        return ParamPack::new_empty();
    return param.ref();
}
__attribute__((visibility("protected")))
PyMethodDef parampack_get_param_method = py::meth_o<"get_param",get_param>;

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&ParamPack::Type) < 0);
}

}
