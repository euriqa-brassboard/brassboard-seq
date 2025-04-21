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

#include "rtprop.h"

#include "rtval.h"

namespace brassboard_seq::rtprop {

using namespace brassboard_seq::rtval;

namespace {

struct composite_rtprop_data : PyObject {
    py::ref<> ovr;
    py::ref<> cache;
    bool compiled;
    bool filled;

    static PyTypeObject Type;
};
PyTypeObject composite_rtprop_data::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.composite_rtprop_data",
    .tp_basicsize = sizeof(composite_rtprop_data),
    .tp_dealloc = py::tp_dealloc<true,[] (py::ptr<composite_rtprop_data> self) {
        call_destructor(&self->ovr);
        call_destructor(&self->cache);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_traverse<[] (py::ptr<composite_rtprop_data> self, auto &visitor) {
        visitor(self->ovr);
        visitor(self->cache);
    }>,
    .tp_clear = py::iunifunc<[] (py::ptr<composite_rtprop_data> self) {
        self->ovr.CLEAR();
        self->cache.CLEAR();
    }>,
};

static inline py::ref<> apply_composite_ovr(py::ptr<> val, py::ptr<> ovr);

static inline bool apply_dict_ovr(py::dict_ref &dict, py::ptr<> k, py::ptr<> v)
{
    if (auto field = dict.try_get(k)) {
        dict.set(k, apply_composite_ovr(field, v));
        return true;
    }
    return false;
}

static inline py::ref<> apply_composite_ovr(py::ptr<> val, py::ptr<> ovr)
{
    if (!ovr.isa<py::dict>())
        return ovr.ref();
    if (!PyDict_Size(ovr))
        return val.ref();
    if (auto d = py::cast<py::dict>(val)) {
        auto newval = d.copy();
        for (auto [k, v]: py::dict_iter(ovr)) {
            if (apply_dict_ovr(newval, k, v))
                continue;
            // for scangroup support since only string key is supported
            if (auto ik = k.try_int(); ik && apply_dict_ovr(newval, ik, v))
                continue;
            newval.set(k, v);
        }
        return newval;
    }
    if (auto l = py::cast<py::list>(val)) {
        auto newval = l.list(); // copy
        for (auto [k, v]: py::dict_iter(ovr)) {
            // for scangroup support since only string key is supported
            auto idx = k.int_().as_int();
            if (idx < 0 || idx >= newval.size())
                py_throw_format(PyExc_IndexError, "list index out of range");
            newval.set(idx, apply_composite_ovr(newval.get(idx), v));
        }
        return newval;
    }
    py_throw_format(PyExc_TypeError, "Unknown value type '%S'", val.type());
}

static inline bool _object_compiled(py::ptr<> obj)
{
    return obj.try_attr("_bb_rt_values"_py).is_none();
}

struct CompositeRTProp : PyObject {
    py::str_ref fieldname;
    py::ref<> cb;

    py::ref<composite_rtprop_data> get_data(py::ptr<> obj)
    {
        if (fieldname == Py_None)
            py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
        if (auto val = obj.try_attr(fieldname);
            val && val.typeis(&composite_rtprop_data::Type))
            return val;
        auto data = py::generic_alloc<composite_rtprop_data>();
        call_constructor(&data->ovr, py::immref(Py_None));
        call_constructor(&data->cache, py::immref(Py_None));
        obj.set_attr(fieldname, data);
        return data;
    }

    py::ref<> get_res(py::ptr<> obj)
    {
        auto data = get_data(obj);
        if (!data->filled || (!data->compiled && _object_compiled(obj))) {
            data->cache.take(cb(obj));
            data->filled = true;
            data->compiled = _object_compiled(obj);
        }
        if (data->ovr == Py_None)
            return data->cache.ref();
        return apply_composite_ovr(data->cache, data->ovr);
    }

    static auto get_state(py::ptr<CompositeRTProp> self, py::ptr<> obj)
    {
        return self->get_data(obj)->ovr.ref();
    }
    static void set_state(py::ptr<CompositeRTProp> self,
                          PyObject *const *args, Py_ssize_t nargs)
    {
        py::check_num_arg("CompositeRTProp.set_state", nargs, 2, 2);
        self->get_data(args[0])->ovr.assign(args[1]);
    }

    static void set_name(py::ptr<CompositeRTProp> self,
                         PyObject *const *args, Py_ssize_t nargs)
    {
        py::check_num_arg("CompositeRTProp.__set_name__", nargs, 2, 2);
        self->fieldname.take("__CompositeRTProp__"_py.concat(args[1]));
    }

    static PyTypeObject Type;
};

PyTypeObject CompositeRTProp::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.CompositeRTProp",
    .tp_basicsize = sizeof(CompositeRTProp),
    .tp_dealloc = py::tp_dealloc<true,[] (py::ptr<CompositeRTProp> self) {
        call_destructor(&self->fieldname);
        call_destructor(&self->cb);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_traverse<[] (py::ptr<CompositeRTProp> self, auto &visitor) {
        visitor(self->cb);
    }>,
    .tp_clear = py::iunifunc<[] (py::ptr<CompositeRTProp> self) {
        self->fieldname.CLEAR();
        self->cb.CLEAR();
    }>,
    .tp_methods = (py::meth_table<
                   py::meth_o<"get_state",get_state>,
                   py::meth_fast<"set_state",set_state>,
                   py::meth_fast<"__set_name__",set_name>>),
    .tp_descr_get = py::trifunc<[] (py::ptr<CompositeRTProp> self, py::ptr<> obj,
                                    auto) -> py::ref<> {
        if (!obj) [[unlikely]]
            return self.ref();
        return self->get_res(obj);
    }>,
    .tp_vectorcall = py::vectorfunc<[] (PyObject*, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_no_kwnames("CompositeRTProp.__init__", kwnames);
        py::check_num_arg("CompositeRTProp.__init__", nargs, 1, 1);
        auto cb = args[0];
        auto self = py::generic_alloc<CompositeRTProp>();
        call_constructor(&self->fieldname, py::immref(Py_None));
        call_constructor(&self->cb, py::newref(cb));
        return self;
    }>,
};

}

namespace {

#define RTPROP_PREFIX_STR "_RTProp_value_"
static constexpr int rtprop_prefix_len = sizeof(RTPROP_PREFIX_STR) - 1;

struct rtprop_callback : ExternCallback {
    py::ref<> obj;
    py::str_ref fieldname;

    static TagVal callback(rtprop_callback *self, unsigned age)
    {
        auto v = self->obj.attr(self->fieldname);
        if (!is_rtval(v))
            return TagVal::from_py(v);
        auto rv = (RuntimeValue*)v;
        if (rv->type_ == ExternAge && rv->cb_arg2 == self)
            py_throw_format(PyExc_ValueError, "RT property have not been assigned.");
        rt_eval_cache(rv, age);
        return rtval_cache(rv);
    }

    static inline py::ref<rtprop_callback> alloc(py::ptr<> obj, py::ptr<> fieldname)
    {
        auto self = py::generic_alloc<rtprop_callback>();
        self->fptr = (void*)callback;
        call_constructor(&self->obj, py::newref(obj));
        call_constructor(&self->fieldname, py::newref(fieldname));
        return self;
    }
    static PyTypeObject Type;
};

PyTypeObject rtprop_callback::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.rtprop_callback",
    .tp_basicsize = sizeof(rtprop_callback),
    .tp_dealloc = py::tp_dealloc<true,[] (py::ptr<rtprop_callback> self) {
        call_destructor(&self->obj);
        call_destructor(&self->fieldname);
    }>,
    .tp_str = py::unifunc<[] (py::ptr<rtprop_callback> self) {
        return py::str_format(
            "<RTProp %U for %S>",
            py::str_ref::checked(PyUnicode_Substring(self->fieldname.get(),
                                                     rtprop_prefix_len,
                                                     PY_SSIZE_T_MAX)), self->obj);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_traverse<[] (py::ptr<rtprop_callback> self, auto &visitor) {
        visitor(self->obj);
    }>,
    .tp_clear = py::iunifunc<[] (py::ptr<rtprop_callback> self) {
        self->obj.CLEAR();
        self->fieldname.CLEAR();
    }>,
    .tp_base = &ExternCallback::Type,
};

struct RTProp : PyObject {
    py::str_ref fieldname;

    py::ref<> get_res(py::ptr<> obj)
    {
        if (fieldname == Py_None)
            py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
        if (auto res = obj.try_attr(fieldname))
            return res;
        py::ref val(new_extern_age(rtprop_callback::alloc(obj, fieldname), &PyFloat_Type));
        obj.set_attr(fieldname, val);
        return val;
    }

    void set_res(py::ptr<> obj, py::ptr<> val)
    {
        if (fieldname == Py_None)
            py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
        if (val && val != Py_None)
            obj.set_attr(fieldname, val);
        else
            obj.del_attr(fieldname);
    }

    static PyObject *get_state(py::ptr<RTProp> self, py::ptr<> obj)
    {
        if (auto res = obj.try_attr(self->fieldname))
            return res.rel();
        Py_RETURN_NONE;
    }

    static void set_state(py::ptr<RTProp> self, PyObject *const *args, Py_ssize_t nargs)
    {
        py::check_num_arg("RTProp.set_state", nargs, 2, 2);
        auto obj = py::ptr(args[0]);
        auto val = args[1];
        if (val == Py_None)
            obj.del_attr(self->fieldname);
        else
            obj.set_attr(self->fieldname, val);
    }

    static void set_name(py::ptr<RTProp> self, PyObject *const *args, Py_ssize_t nargs)
    {
        py::check_num_arg("RTProp.__set_name__", nargs, 2, 2);
        self->fieldname.take(RTPROP_PREFIX_STR ""_py.concat(args[1]));
    }

    static PyTypeObject Type;
};

PyTypeObject RTProp::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.RTProp",
    .tp_basicsize = sizeof(RTProp),
    .tp_dealloc = py::tp_dealloc<false,[] (py::ptr<RTProp> self) {
        call_destructor(&self->fieldname);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (py::meth_table<
                   py::meth_o<"get_state",get_state>,
                   py::meth_fast<"set_state",set_state>,
                   py::meth_fast<"__set_name__",set_name>>),
    .tp_descr_get = py::trifunc<[] (py::ptr<RTProp> self, py::ptr<> obj,
                                    auto) -> py::ref<> {
        if (!obj) [[unlikely]]
            return self.ref();
        return self->get_res(obj);
    }>,
    .tp_descr_set = py::itrifunc<[] (py::ptr<RTProp> self, py::ptr<> obj, py::ptr<> val) {
        self->set_res(obj, val);
    }>,
    .tp_vectorcall = py::vectorfunc<[] (PyObject*, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_no_kwnames("RTProp.__init__", kwnames);
        py::check_num_arg("RTProp.__init__", nargs, 0, 0);
        auto self = py::generic_alloc<RTProp>();
        call_constructor(&self->fieldname, py::immref(Py_None));
        return self;
    }>,
};

}

__attribute__((visibility("protected")))
PyTypeObject &CompositeRTProp_Type = CompositeRTProp::Type;
__attribute__((visibility("protected")))
PyTypeObject &RTProp_Type = RTProp::Type;

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&composite_rtprop_data::Type) < 0);
    throw_if(PyType_Ready(&CompositeRTProp::Type) < 0);
    throw_if(PyType_Ready(&rtprop_callback::Type) < 0);
    throw_if(PyType_Ready(&RTProp::Type) < 0);
}

}
