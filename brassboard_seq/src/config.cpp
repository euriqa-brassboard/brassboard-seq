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

static inline void add_supported_prefix(py::ptr<Config> self, py::ptr<> prefix)
{
    py::set(self->supported_prefix).add(py::arg_cast<py::str>(prefix, "prefix"));
}

static inline void add_channel_alias(py::ptr<Config> self, PyObject *const *args,
                                     Py_ssize_t nargs)
{
    py::check_num_arg("Config.add_channel_alias", nargs, 2, 2);
    auto name = py::arg_cast<py::str>(args[0], "name");
    auto target = py::arg_cast<py::str>(args[1], "target");
    if (py::str(name).contains("/"_py))
        py_throw_format(PyExc_ValueError, "Channel alias name may not contain \"/\"");
    py::dict(self->alias_cache).clear();
    py::dict(self->channel_alias).set(name, split_string_tuple(target));
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

static inline PyObject *py_translate_channel(py::ptr<Config> self, py::ptr<> name)
{
    return self->translate_channel(py::arg_cast<py::str>(name, "name"));
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
    .tp_methods = (py::meth_table<
                   py::meth_o<"add_supported_prefix",add_supported_prefix>,
                   py::meth_fast<"add_channel_alias",add_channel_alias>,
                   py::meth_o<"translate_channel",py_translate_channel>>),
    .tp_new = py::tp_new<[] (PyTypeObject *t, auto...) {
        auto self = py::generic_alloc<Config>(t);
        self->channel_alias = py::new_dict().rel();
        self->alias_cache = py::new_dict().rel();
        self->supported_prefix = py::new_set().rel();
        return self;
    }>,
};

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&Config::Type) < 0);
}

}
