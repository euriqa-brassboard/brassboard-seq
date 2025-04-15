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

#include "config.h"

namespace brassboard_seq::config {

static inline void check_string_arg(py::ptr<> arg, const char *name)
{
    if (arg.isa<py::str>())
        return;
    py_throw_format(PyExc_TypeError, "%s must be a string", name);
}

static inline auto split_string_tuple(PyObject *s)
{
    auto list = py::list_ref(throw_if_not(PyUnicode_Split(s, "/"_py, -1)));
    auto tuple = py::new_tuple(list.size());
    for (auto [i, v]: py::list_iter(list)) {
        tuple.SET(i, py::ref(v.get())); // Steal reference
        list.SET(i, nullptr);
    }
    return tuple;
}

static PyObject *add_supported_prefix(Config *self, PyObject *const *args,
                                      Py_ssize_t nargs)
{
    return cxx_catch([&] {
        py_check_num_arg("add_supported_prefix", nargs, 1, 1);
        check_string_arg(args[0], "prefix");
        py::set(self->supported_prefix).add(args[0]);
        Py_RETURN_NONE;
    });
}

static PyObject *add_channel_alias(Config *self, PyObject *const *args,
                                   Py_ssize_t nargs)
{
    return cxx_catch([&] {
        py_check_num_arg("add_channel_alias", nargs, 2, 2);
        auto name = args[0];
        auto target = args[1];
        check_string_arg(name, "name");
        check_string_arg(target, "target");
        if (py::str(name).contains("/"_py))
            py_throw_format(PyExc_ValueError, "Channel alias name may not contain \"/\"");
        py::dict(self->alias_cache).clear();
        py::dict(self->channel_alias).set(name, split_string_tuple(target));
        Py_RETURN_NONE;
    });
}

inline py::tuple_ref Config::_translate_channel(py::tuple path)
{
    auto alias_cache_dict = py::dict(alias_cache);
    if (auto resolved = alias_cache_dict.try_get(path))
        return resolved.ref();
    // Hardcoded limit for loop detection
    auto path_len = path.size();
    if (path_len > 10)
        py_throw_format(PyExc_ValueError, "Channel alias loop detected: %U",
                        channel_name_from_path(path));
    auto prefix = path.get(0);
    auto channel_alias_dict = py::dict(channel_alias);
    if (auto new_prefix = py::tuple(channel_alias_dict.try_get(prefix))) {
        auto prefix_len = new_prefix.size();
        auto newpath = py::new_tuple(prefix_len + path_len - 1);
        for (auto [i, v]: py::tuple_iter(new_prefix))
            newpath.SET(i, v);
        for (auto [i, v]: py::tuple_iter(path))
            if (i != 0)
                newpath.SET(i + prefix_len - 1, v);
        auto resolved = _translate_channel(newpath);
        py::dict(alias_cache).set(path, resolved);
        return resolved;
    }
    if (!py::set(supported_prefix).contains(prefix))
        py_throw_format(PyExc_ValueError, "Unsupported channel name: %U",
                        channel_name_from_path(path));
    py::dict(alias_cache).set(path, path);
    return path.ref();
}

__attribute__((visibility("protected")))
PyObject *Config::translate_channel(PyObject *name)
{
    return _translate_channel(split_string_tuple(name)).rel();
}

static PyObject *py_translate_channel(Config *self, PyObject *const *args,
                                      Py_ssize_t nargs)
{
    return cxx_catch([&] {
        py_check_num_arg("translate_channel", nargs, 1);
        check_string_arg(args[0], "name");
        return self->translate_channel(args[0]);
    });
}

__attribute__((visibility("protected")))
PyTypeObject Config::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.config.Config",
    .tp_basicsize = sizeof(Config),
    .tp_dealloc = [] (PyObject *py_self) {
        auto self = (Config*)py_self;
        Py_CLEAR(self->channel_alias);
        Py_CLEAR(self->alias_cache);
        Py_CLEAR(self->supported_prefix);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    // All fields are containers of immutable types.
    // No reference loop possible.
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,
    .tp_methods = (PyMethodDef[]){
        {"add_supported_prefix", (PyCFunction)(void*)add_supported_prefix,
         METH_FASTCALL, 0},
        {"add_channel_alias", (PyCFunction)(void*)add_channel_alias, METH_FASTCALL, 0},
        {"translate_channel", (PyCFunction)(void*)py_translate_channel, METH_FASTCALL, 0},
        {0, 0, 0, 0}
    },
    .tp_new = [] (PyTypeObject *t, PyObject*, PyObject*) -> PyObject* {
        return cxx_catch([&] {
            auto self = py::generic_alloc<Config>(t);
            self->channel_alias = py::new_dict().rel();
            self->alias_cache = py::new_dict().rel();
            self->supported_prefix = py::new_set().rel();
            return self;
        });
    },
};

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&Config::Type) < 0);
}

}
