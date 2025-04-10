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

static inline void check_string_arg(PyObject *arg, const char *name)
{
    if (PyUnicode_CheckExact(arg))
        return;
    py_throw_format(PyExc_TypeError, "%s must be a string", name);
}

static inline PyObject *split_string_tuple(PyObject *s)
{
    auto list = pyobj_checked(PyUnicode_Split(s, "/"_py, -1));
    py_object tuple(pytuple_new(PyList_GET_SIZE(list.get())));
    for (auto [i, v]: pylist_iter(list)) {
        PyTuple_SET_ITEM(tuple.get(), i, v);
        PyList_SET_ITEM(list.get(), i, nullptr);
    }
    return tuple.release();
}

static PyObject *add_supported_prefix(Config *self, PyObject *const *args,
                                      Py_ssize_t nargs)
{
    return cxx_catch([&] {
        py_check_num_arg("add_supported_prefix", nargs, 1, 1);
        check_string_arg(args[0], "prefix");
        throw_if(PySet_Add(self->supported_prefix, args[0]));
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
        if (py_check_int(PyUnicode_Contains(name, "/"_py)))
            py_throw_format(PyExc_ValueError, "Channel alias name may not contain \"/\"");
        PyDict_Clear(self->alias_cache);
        pydict_setitem(self->channel_alias, name, py_object(split_string_tuple(target)));
        Py_RETURN_NONE;
    });
}

inline PyObject *Config::_translate_channel(PyObject *path)
{
    if (auto resolved = PyDict_GetItemWithError(alias_cache, path))
        return py_newref(resolved);
    throw_pyerr();
    // Hardcoded limit for loop detection
    if (PyTuple_GET_SIZE(path) > 10)
        py_throw_format(PyExc_ValueError, "Channel alias loop detected: %U",
                        channel_name_from_path(path).get());
    auto prefix = PyTuple_GET_ITEM(path, 0);
    if (auto new_prefix = PyDict_GetItemWithError(channel_alias, prefix)) {
        auto prefix_len = PyTuple_GET_SIZE(new_prefix);
        py_object newpath(pytuple_new(prefix_len + PyTuple_GET_SIZE(path) - 1));
        for (auto [i, v]: pytuple_iter(new_prefix))
            PyTuple_SET_ITEM(newpath.get(), i, py_newref(v));
        for (auto [i, v]: pytuple_iter(path))
            if (i != 0)
                PyTuple_SET_ITEM(newpath.get(), i + prefix_len - 1, py_newref(v));
        py_object resolved(_translate_channel(newpath));
        pydict_setitem(alias_cache, path, resolved);
        return resolved.release();
    }
    throw_pyerr();
    if (!py_check_int(PySet_Contains(supported_prefix, prefix)))
        py_throw_format(PyExc_ValueError, "Unsupported channel name: %U",
                        channel_name_from_path(path).get());
    pydict_setitem(alias_cache, path, path);
    return py_newref(path);
}

__attribute__((visibility("protected")))
PyObject *Config::translate_channel(PyObject *name)
{
    return _translate_channel(py_object(split_string_tuple(name)));
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
            auto py_self = pytype_genericalloc(t);
            auto self = (Config*)py_self;
            self->channel_alias = pydict_new();
            self->alias_cache = pydict_new();
            self->supported_prefix = pyset_new();
            return py_self;
        });
    },
};

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&Config::Type) < 0);
}

}
