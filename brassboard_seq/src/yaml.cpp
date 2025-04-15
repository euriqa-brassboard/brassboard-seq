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

#include "yaml.h"

#include "numpy/arrayobject.h"

namespace brassboard_seq::yaml {

static void print_generic(py::stringio &io, PyObject *obj, int indent, int cur_indent);

static inline bool needs_quote(PyObject *s)
{
    // Good enough for now........
    auto len = PyUnicode_GET_LENGTH(s);
    for (auto [i, c]: py::str_iter(s)) {
        if (c == ' ' && (i == 0 || i == len - 1))
            return true;
        if (c == '"' || c == ':' || c == '\n' || c == '\b' || c == '\\') {
            return true;
        }
    }
    return false;
}

static inline void print_string(py::stringio &io, PyObject *s, int indent, int cur_indent)
{
    auto orig_len = PyUnicode_GET_LENGTH(s);
    if (!orig_len)
        return io.write_ascii("\"\"");
    auto write_str_prefix = [&] (auto len) {
        if (indent < cur_indent && cur_indent + len > 85) {
            io.write_ascii("\n");
            io.write_rep_ascii(indent, " ");
        }
    };
    if (needs_quote(s)) {
        auto quoted_len = orig_len + 2;
        for (auto [i, c]: py::str_iter(s)) {
            if (c == '"' || c == '\n' || c == '\b' || c == '\\') {
                quoted_len += 1;
            }
        }
        write_str_prefix(quoted_len);
        int k = PyUnicode_KIND(s);
        auto [buff_kind, buff] = io.reserve_buffer(k, quoted_len);
        PyUnicode_WRITE(buff_kind, buff, 0, '"');
        Py_ssize_t offset = 1;
        for (auto [i, c]: py::str_iter(s)) {
            switch (c) {
            case '"':
                PyUnicode_WRITE(buff_kind, buff, i + offset, '\\');
                PyUnicode_WRITE(buff_kind, buff, i + (++offset), '"');
                break;
            case '\n':
                PyUnicode_WRITE(buff_kind, buff, i + offset, '\\');
                PyUnicode_WRITE(buff_kind, buff, i + (++offset), 'n');
                break;
            case '\b':
                PyUnicode_WRITE(buff_kind, buff, i + offset, '\\');
                PyUnicode_WRITE(buff_kind, buff, i + (++offset), 'b');
                break;
            case '\\':
                PyUnicode_WRITE(buff_kind, buff, i + offset, '\\');
                PyUnicode_WRITE(buff_kind, buff, i + (++offset), '\\');
                break;
            default:
                PyUnicode_WRITE(buff_kind, buff, i + offset, c);
            }
        }
        PyUnicode_WRITE(buff_kind, buff, quoted_len - 1, '"');
    }
    else {
        write_str_prefix(orig_len);
        io.write(s);
    }
}

static inline void print_single_field_dict(py::stringio &io, PyObject *obj, int indent,
                                           int cur_indent, PyObject *prefix_name=nullptr)
{
    // `indent < cur_indent` can only happen if `print_generic` is called with
    // this condition following the call chain:
    // `print_generic -> print_scalar -> print_dict` which in turns can only happen in
    // `print_dict_field` or `print_single_field_dict` where the object cannot be
    // a single field dict.
    assert(indent >= cur_indent);

    cur_indent = indent + 1;
    bool isfirst = true;
    if (prefix_name) {
        cur_indent += PyUnicode_GET_LENGTH(prefix_name) + 1;
        io.write(prefix_name);
        isfirst = false;
    }
    while (PyDict_Check(obj)) {
        if (PyDict_GET_SIZE(obj) != 1)
            break;
        for (auto [k, v]: py::dict_iter(obj)) {
            if (!PyUnicode_Check(k))
                py_throw_format(PyExc_TypeError, "yaml dict key must be str");
            obj = v;
            if (!isfirst)
                io.write_ascii(".");
            isfirst = false;
            cur_indent += PyUnicode_GET_LENGTH(k) + 1;
            io.write(k);
        }
    }
    io.write_ascii(":");
    py::stringio io2;
    print_generic(io2, obj, indent + 2, cur_indent);
    auto strfield = io2.getvalue();
    assert(strfield.size());
    if (PyUnicode_READ_CHAR(strfield.get(), 0) != '\n')
        io.write_ascii(" ");
    io.write(strfield);
}

static inline void print_dict_field(py::stringio &io, PyObject *k, PyObject *v, int indent)
{
    if (PyDict_Check(v) && PyDict_GET_SIZE(v) == 1)
        return print_single_field_dict(io, v, indent, indent, k);
    auto keylen = PyUnicode_GET_LENGTH(k);
    io.write(k);
    io.write_ascii(":");
    py::stringio io2;
    print_generic(io2, v, indent + 2, indent + 2 + keylen);
    auto strfield = io2.getvalue();
    assert(strfield.size());
    if (PyUnicode_READ_CHAR(strfield.get(), 0) != '\n')
        io.write_ascii(" ");
    io.write(strfield);
}

static inline void print_dict(py::stringio &io, PyObject *obj, int indent, int cur_indent)
{
    int nmembers = PyDict_GET_SIZE(obj);
    if (nmembers == 0) {
        return io.write_ascii("{}");
    }
    else if (nmembers == 1) {
        return print_single_field_dict(io, obj, indent, cur_indent);
    }
    if (indent < cur_indent) {
        io.write_ascii("\n");
        io.write_rep_ascii(indent, " ");
    }
    bool isfirst = true;
    for (auto [k, v]: py::dict_iter(obj)) {
        if (!isfirst) {
            io.write_ascii("\n");
            io.write_rep_ascii(indent, " ");
        }
        isfirst = false;
        print_dict_field(io, k, v, indent);
    }
}

static inline bool is_bool_obj(PyObject *obj)
{
    return PyBool_Check(obj) || PyArray_IsScalar(obj, Bool);
}

static inline void print_scalar(py::stringio &io, PyObject *obj, int indent, int cur_indent)
{
    if (is_bool_obj(obj))
        return io.write_ascii(get_value_bool(obj, -1) ? "true" : "false");
    if (PyUnicode_Check(obj))
        return print_string(io, obj, indent, cur_indent);
    if (PyDict_Check(obj))
        return print_dict(io, obj, indent, cur_indent);
    if (PyArray_IsPythonNumber(obj) || PyArray_IsScalar(obj, Number))
        return io.write_str(obj);
    io.write_ascii("<unknown object ");
    io.write_str(obj);
    io.write_ascii(">");
}

static void print_array_str(py::stringio &io, std::vector<py::str_ref> &strary,
                            bool all_short_scalar, int indent, int cur_indent)
{
    if (all_short_scalar) {
        auto threshold = std::max(85 - cur_indent, 50);
        auto single_line = indent < cur_indent;
        int linelen = 0;
        auto nele = strary.size();
        bool prefix_printed = false;
        size_t print_idx = 0;
        auto ensure_prefix = [&] {
            if (!prefix_printed) {
                io.write_ascii("[");
                prefix_printed = true;
            }
        };
        auto print_list_until = [&] (size_t i) {
            bool isfirst = true;
            for (; print_idx < i; print_idx++) {
                if (!isfirst)
                    io.write_ascii(", ");
                isfirst = false;
                io.write(strary[print_idx]);
            }
        };
        for (size_t i = 0; i < nele; i++) {
            if (linelen > threshold) {
                if (single_line) {
                    single_line = false;
                    threshold = std::max(85 - indent, 50);
                    assert(!prefix_printed);
                    io.write_ascii("\n");
                    io.write_rep_ascii(indent, " ");
                }
                ensure_prefix();
                if (linelen > threshold && i < nele - 1) {
                    print_list_until(i);
                    io.write_ascii(",\n");
                    io.write_rep_ascii(indent + 1, " ");
                    linelen = 0;
                }
            }
            linelen += strary[i].size() + 2;
        }
        ensure_prefix();
        print_list_until(nele);
        return io.write_ascii("]");
    }
    bool isfirst = true;
    for (auto &v: strary) {
        if (!isfirst || indent < cur_indent) {
            io.write_ascii("\n");
            io.write_rep_ascii(indent, " ");
        }
        isfirst = false;
        io.write_ascii("- ");
        io.write(v);
    }
}

static inline void print_array_iter(py::stringio &io, auto &&iter,
                                    int indent, int cur_indent)
{
    std::vector<py::str_ref> strary;
    bool all_short_scalar = true;
    for (auto [i, v]: iter) {
        if (is_bool_obj(v)) {
            strary.push_back((get_value_bool(v, -1) ? "true"_py : "false"_py).ref());
        }
        else if (PyUnicode_Check(v)) {
            py::stringio io2;
            print_string(io2, v, 0, 0);
            auto s = io2.getvalue();
            if (s.size() > 16)
                all_short_scalar = false;
            strary.push_back(std::move(s));
        }
        else if (PyArray_IsPythonNumber(v) || PyArray_IsScalar(v, Number)) {
            strary.push_back(v.str());
        }
        else if (PyDict_Check(v) && PyDict_GET_SIZE(v) == 0) {
            strary.push_back("{}"_py.ref());
        }
        else if ((PyList_Check(v) && PyList_GET_SIZE(v) == 0) ||
                 (PyTuple_Check(v) && PyTuple_GET_SIZE(v) == 0) ||
                 (PyArray_Check(v) && PyArray_NDIM((PyArrayObject*)v) == 1 &&
                  PyArray_DIM((PyArrayObject*)v, 0) == 0)) {
            strary.push_back("[]"_py.ref());
        }
        else {
            py::stringio io2;
            print_generic(io2, v, indent + 2, indent + 2);
            strary.push_back(io2.getvalue());
            all_short_scalar = false;
        }
    }
    print_array_str(io, strary, all_short_scalar, indent, cur_indent);
}

static inline void print_array_numpy(py::stringio &io, PyArrayObject *ary,
                                     int indent, int cur_indent)
{
    std::vector<py::str_ref> strary;
    auto data = (char*)PyArray_DATA(ary);
    auto sz = PyArray_DIM(ary, 0);
    auto elsz = PyArray_ITEMSIZE(ary);
    if (PyArray_ISBOOL(ary)) {
        bool *values = (bool*)data;
        for (int i = 0; i < sz; i++) {
            strary.push_back((values[i] ? "true"_py : "false"_py).ref());
        }
    }
    else {
        for (int i = 0; i < sz; i++) {
            auto item = py::ref<>::checked(PyArray_GETITEM(ary, data + i * elsz));
            strary.push_back(item.str());
        }
    }
    print_array_str(io, strary, true, indent, cur_indent);
}

static void print_generic(py::stringio &io, PyObject *obj, int indent, int cur_indent)
{
    if (PyList_Check(obj))
        return print_array_iter(io, py::list_iter(obj), indent, cur_indent);
    if (PyTuple_Check(obj))
        return print_array_iter(io, py::tuple_iter(obj), indent, cur_indent);
    if (PyArray_Check(obj)) {
        if (PyArray_NDIM((PyArrayObject*)obj) != 1)
            py_throw_format(PyExc_TypeError, "yaml only support ndarray of dimension 1");
        return print_array_numpy(io, (PyArrayObject*)obj, indent, cur_indent);
    }
    return print_scalar(io, obj, indent, cur_indent);
}

__attribute__((visibility("protected")))
void print(py::stringio &io, PyObject *obj, int indent)
{
    print_generic(io, obj, indent, indent);
}

__attribute__((visibility("protected")))
PyObject *sprint(PyObject *obj, int indent)
{
    py::stringio io;
    print_generic(io, obj, indent, indent);
    return io.getvalue().rel();
}

static PyObject *py_sprint(PyObject*, PyObject *const *args, Py_ssize_t nargs)
{
    return cxx_catch([&] {
        py_check_num_arg("sprint", nargs, 1, 2);
        int indent = 0;
        if (nargs >= 2) {
            if (!PyLong_Check(args[1]))
                py_throw_format(PyExc_TypeError, "indent must be integer");
            indent = PyLong_AsLong(args[1]);
            if (indent < 0) {
                throw_pyerr();
                py_throw_format(PyExc_TypeError, "indent cannot be negative");
            }
        }
        return sprint(args[0], indent);
    });
}

PyMethodDef sprint_method = {"sprint", (PyCFunction)(void*)py_sprint,
    METH_FASTCALL, 0};

__attribute__((constructor))
static void init()
{
    _import_array();
}

}
