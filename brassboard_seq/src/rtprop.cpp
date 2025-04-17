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
    PyObject *ovr;
    PyObject *cache;
    bool compiled;
    bool filled;

    static PyTypeObject Type;
};
PyTypeObject composite_rtprop_data::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.composite_rtprop_data",
    .tp_basicsize = sizeof(composite_rtprop_data),
    .tp_dealloc = [] (PyObject *py_self) {
        PyObject_GC_UnTrack(py_self);
        Type.tp_clear(py_self);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        auto self = (composite_rtprop_data*)py_self;
        Py_VISIT(self->ovr);
        Py_VISIT(self->cache);
        return 0;
    },
    .tp_clear = [] (PyObject *py_self) {
        auto self = (composite_rtprop_data*)py_self;
        Py_CLEAR(self->ovr);
        Py_CLEAR(self->cache);
        return 0;
    },
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
            auto idx = PyLong_AsLong(k.int_().get());
            if (idx < 0) {
                throw_pyerr();
                py_throw_format(PyExc_IndexError, "list index out of range");
            }
            if (idx >= newval.size())
                py_throw_format(PyExc_IndexError, "list index out of range");
            newval.set(idx, apply_composite_ovr(newval.get(idx), v));
        }
        return newval;
    }
    py_throw_format(PyExc_TypeError, "Unknown value type '%S'", Py_TYPE(val));
}

static inline bool _object_compiled(py::ptr<> obj)
{
    return obj.try_attr("_bb_rt_values"_py).is_none();
}

struct CompositeRTProp : PyObject {
    PyObject *fieldname;
    PyObject *cb;

    __attribute__((alias("_ZN14brassboard_seq6rtprop20CompositeRTProp_TypeE")))
    static PyTypeObject Type;

    py::ref<composite_rtprop_data> get_data(py::ptr<> obj)
    {
        if (fieldname == Py_None)
            py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
        if (auto val = obj.try_attr(fieldname);
            val && val.typeis(&composite_rtprop_data::Type))
            return val;
        auto data = py::generic_alloc<composite_rtprop_data>();
        data->ovr = py::immref(Py_None);
        data->cache = py::immref(Py_None);
        obj.set_attr(fieldname, data);
        return data;
    }

    py::ref<> get_res(py::ptr<> obj)
    {
        auto data = get_data(obj);
        if (!data->filled || (!data->compiled && _object_compiled(obj))) {
            py::assign(data->cache, py::ptr(cb)(obj));
            data->filled = true;
            data->compiled = _object_compiled(obj);
        }
        if (data->ovr == Py_None)
            return py::ptr(data->cache).ref();
        return apply_composite_ovr(data->cache, data->ovr);
    }

    static PyObject *get_state(py::ptr<CompositeRTProp> self, py::ptr<> obj)
    {
        return py::newref(self->get_data(obj)->ovr);
    }
    static void set_state(py::ptr<CompositeRTProp> self,
                          PyObject *const *args, Py_ssize_t nargs)
    {
        py::check_num_arg("CompositeRTProp.set_state", nargs, 2, 2);
        py::assign(self->get_data(args[0])->ovr, args[1]);
    }

    static void set_name(py::ptr<CompositeRTProp> self,
                         PyObject *const *args, Py_ssize_t nargs)
    {
        py::check_num_arg("CompositeRTProp.__set_name__", nargs, 2, 2);
        py::assign(self->fieldname, "__CompositeRTProp__"_py.concat(args[1]));
    }
};

}

__attribute__((visibility("protected")))
PyTypeObject CompositeRTProp_Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.CompositeRTProp",
    .tp_basicsize = sizeof(CompositeRTProp),
    .tp_dealloc = [] (PyObject *py_self) {
        PyObject_GC_UnTrack(py_self);
        CompositeRTProp_Type.tp_clear(py_self);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        auto self = (CompositeRTProp*)py_self;
        Py_VISIT(self->cb);
        return 0;
    },
    .tp_clear = [] (PyObject *py_self) {
        auto self = (CompositeRTProp*)py_self;
        Py_CLEAR(self->fieldname);
        Py_CLEAR(self->cb);
        return 0;
    },
    .tp_methods = (py::meth_table<
                   py::meth_o<"get_state",CompositeRTProp::get_state>,
                   py::meth_fast<"set_state",CompositeRTProp::set_state>,
                   py::meth_fast<"__set_name__",CompositeRTProp::set_name>>),
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
        auto data = py::generic_alloc<CompositeRTProp>();
        data->fieldname = py::immref(Py_None);
        data->cb = py::newref(cb);
        return data;
    }>,
};

namespace {

#define RTPROP_PREFIX_STR "_RTProp_value_"
static constexpr int rtprop_prefix_len = sizeof(RTPROP_PREFIX_STR) - 1;

struct rtprop_callback : ExternCallback {
    PyObject *obj;
    PyObject *fieldname;

    static TagVal callback(rtprop_callback *self, unsigned age)
    {
        auto v = py::ptr(self->obj).attr(self->fieldname);
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
        self->obj = py::newref(obj);
        self->fieldname = py::newref(fieldname);
        return self;
    }
    static PyTypeObject Type;
};

PyTypeObject rtprop_callback::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.rtprop_callback",
    .tp_basicsize = sizeof(rtprop_callback),
    .tp_dealloc = [] (PyObject *py_self) {
        PyObject_GC_UnTrack(py_self);
        Type.tp_clear(py_self);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    .tp_str = py::unifunc<[] (py::ptr<rtprop_callback> self) {
        return py::str_format(
            "<RTProp %U for %S>",
            py::str_ref::checked(PyUnicode_Substring(self->fieldname, rtprop_prefix_len,
                                                     PY_SSIZE_T_MAX)), self->obj);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = [] (PyObject *py_self, visitproc visit, void *arg) {
        auto self = (rtprop_callback*)py_self;
        Py_VISIT(self->obj);
        return 0;
    },
    .tp_clear = [] (PyObject *py_self) {
        auto self = (rtprop_callback*)py_self;
        Py_CLEAR(self->obj);
        Py_CLEAR(self->fieldname);
        return 0;
    },
    .tp_base = &ExternCallback::Type,
};

struct RTProp : PyObject {
    PyObject *fieldname;

    __attribute__((alias("_ZN14brassboard_seq6rtprop11RTProp_TypeE")))
    static PyTypeObject Type;

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
        py::assign(self->fieldname, RTPROP_PREFIX_STR ""_py.concat(args[1]));
    }
};

}

__attribute__((visibility("protected")))
PyTypeObject RTProp_Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.RTProp",
    .tp_basicsize = sizeof(RTProp),
    .tp_dealloc = [] (PyObject *py_self) {
        auto self = (RTProp*)py_self;
        Py_CLEAR(self->fieldname);
        Py_TYPE(py_self)->tp_free(py_self);
    },
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = (py::meth_table<
                   py::meth_o<"get_state",RTProp::get_state>,
                   py::meth_fast<"set_state",RTProp::set_state>,
                   py::meth_fast<"__set_name__",RTProp::set_name>>),
    .tp_descr_get = py::trifunc<[] (py::ptr<RTProp> self, py::ptr<> obj, auto) -> py::ref<> {
        if (!obj) [[unlikely]]
            return self.ref();
        return self->get_res(obj);
    }>,
    .tp_descr_set = py::itrifunc<[] (py::ptr<RTProp> self, py::ptr<> obj, py::ptr<> val) {
        self->set_res(obj, val);
        return 0;
    }>,
    .tp_vectorcall = py::vectorfunc<[] (PyObject*, PyObject *const *args,
                                        ssize_t nargs, py::tuple kwnames) {
        py::check_no_kwnames("RTProp.__init__", kwnames);
        py::check_num_arg("RTProp.__init__", nargs, 0, 0);
        auto self = py::generic_alloc<RTProp>();
        self->fieldname = py::immref(Py_None);
        return self;
    }>,
};

__attribute__((constructor))
static void init()
{
    throw_if(PyType_Ready(&composite_rtprop_data::Type) < 0);
    throw_if(PyType_Ready(&CompositeRTProp_Type) < 0);
    throw_if(PyType_Ready(&rtprop_callback::Type) < 0);
    throw_if(PyType_Ready(&RTProp_Type) < 0);
}

}
