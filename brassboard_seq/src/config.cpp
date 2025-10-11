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

__attribute__((visibility("internal")))
py::str_ref channel_name_from_path(py::ptr<> path)
{
    return "/"_py.join(path);
}

[[noreturn]] __attribute__((visibility("internal")))
void raise_invalid_channel(py::tuple path)
{
    py_throw_format(PyExc_ValueError, "Invalid channel name %U",
                    channel_name_from_path(path));
}

static inline auto split_string_tuple(py::str s)
{
    auto list = s.split("/"_py, -1);
    return py::new_ntuple(list.size(), [&] (int i) {
        auto v = py::ref(list.get(i).get()); // Steal reference
        list.SET(i, nullptr);
        return v;
    });
}

__attribute__((visibility("internal")))
inline py::tuple_ref Config::_translate_channel(py::tuple path)
{
    if (auto resolved = alias_cache.try_get(path))
        return resolved.ref();
    // Hardcoded limit for loop detection
    auto path_len = path.size();
    if (path_len > 10)
        py_throw_format(PyExc_ValueError, "Channel alias loop detected: %U",
                        channel_name_from_path(path));
    auto prefix = path.get(0);
    if (auto new_prefix = py::tuple(channel_alias.try_get(prefix))) {
        auto prefix_len = new_prefix.size();
        auto newpath = py::new_tuple(prefix_len + path_len - 1);
        for (auto [i, v]: py::tuple_iter(new_prefix))
            newpath.SET(i, v);
        for (auto [i, v]: py::tuple_iter(path))
            if (i != 0)
                newpath.SET(i + prefix_len - 1, v);
        auto resolved = _translate_channel(newpath);
        alias_cache.set(path, resolved);
        return resolved;
    }
    if (!supported_prefix.contains(prefix))
        py_throw_format(PyExc_ValueError, "Unsupported channel name: %U",
                        channel_name_from_path(path));
    alias_cache.set(path, path);
    return path.ref();
}

__attribute__((visibility("internal")))
py::tuple_ref Config::translate_channel(py::str name)
{
    return _translate_channel(split_string_tuple(name));
}

BB_PROTECTED
py::ref<Config> Config::alloc(PyTypeObject *t)
{
    auto self = py::generic_alloc<Config>(t);
    call_constructor(&self->channel_alias, py::new_dict());
    call_constructor(&self->alias_cache, py::new_dict());
    call_constructor(&self->supported_prefix, py::new_set());
    return self;
}

BB_PROTECTED
PyTypeObject Config::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.config.Config",
    .tp_basicsize = sizeof(Config),
    .tp_dealloc = py::tp_cxx_dealloc<false,Config>,
    // All fields are containers of immutable types.
    // No reference loop possible.
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,
    .tp_methods = (
        py::meth_table<
        py::meth_o<"add_supported_prefix",[] (py::ptr<Config> self, py::ptr<> prefix) {
            self->supported_prefix.add(py::arg_cast<py::str>(prefix, "prefix"));
        }>,
        py::meth_fast<"add_channel_alias",[] (py::ptr<Config> self, PyObject *const *args,
                                              Py_ssize_t nargs) {
            py::check_num_arg("Config.add_channel_alias", nargs, 2, 2);
            auto name = py::arg_cast<py::str>(args[0], "name");
            auto target = py::arg_cast<py::str>(args[1], "target");
            if (py::str(name).contains("/"_py))
                py_throw_format(PyExc_ValueError, "Channel alias name may not contain \"/\"");
            self->alias_cache.clear();
            self->channel_alias.set(name, split_string_tuple(target));
        }>,
        py::meth_o<"translate_channel",[] (py::ptr<Config> self, py::ptr<> name) {
            return self->translate_channel(py::arg_cast<py::str>(name, "name"));
        }>>),
    .tp_new = py::tp_new<[] (PyTypeObject *t, auto...) {
        return alloc(t);
    }>,
};

__attribute__((visibility("hidden")))
void init()
{
    throw_if(PyType_Ready(&Config::Type) < 0);
}

}
