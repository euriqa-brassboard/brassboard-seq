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

static inline PyObject *apply_composite_ovr(PyObject *val, PyObject *ovr);

static inline bool apply_dict_ovr(PyObject *dict, PyObject *k, PyObject *v)
{
    auto field = PyDict_GetItemWithError(dict, k);
    if (field) {
        throw_if(PyDict_SetItem(dict, k, py_object(apply_composite_ovr(field, v))));
        return true;
    }
    throw_if(PyErr_Occurred());
    return false;
}

static inline PyObject *apply_composite_ovr(PyObject *val, PyObject *ovr)
{
    if (!PyDict_Check(ovr))
        return py_newref(ovr);
    if (!PyDict_Size(ovr))
        return py_newref(val);
    if (PyDict_Check(val)) {
        auto newval = pyobj_checked(PyDict_Copy(val));
        for (auto [k, v]: pydict_iter(ovr)) {
            if (apply_dict_ovr(newval, k, v))
                continue;
            // for scangroup support since only string key is supported
            if (py_object ik(PyNumber_Long(k)); !ik) {
                PyErr_Clear();
            }
            else if (apply_dict_ovr(newval, ik, v)) {
                continue;
            }
            throw_if(PyDict_SetItem(newval, k, v));
        }
        return newval.release();
    }
    if (PyList_Check(val)) {
        auto newval = pyobj_checked(PySequence_List(val));
        for (auto [k, v]: pydict_iter(ovr)) {
            // for scangroup support since only string key is supported
            auto idx = PyLong_AsLong(pyobj_checked(PyNumber_Long(k)).get());
            if (idx < 0) {
                throw_if(PyErr_Occurred());
                py_throw_format(PyExc_IndexError, "list index out of range");
            }
            if (idx >= PyList_GET_SIZE(newval.get()))
                py_throw_format(PyExc_IndexError, "list index out of range");
            auto olditem = PyList_GET_ITEM(newval.get(), idx);
            PyList_SET_ITEM(newval.get(), idx, apply_composite_ovr(olditem, v));
            Py_DECREF(olditem);
        }
        return newval.release();
    }
    py_throw_format(PyExc_TypeError, "Unknown value type '%S'", Py_TYPE(val));
}

static inline bool _object_compiled(PyObject *obj)
{
    py_object field(PyObject_GetAttr(obj, "_bb_rt_values"_py));
    if (!field)
        PyErr_Clear();
    return field.get() == Py_None;
}

struct CompositeRTProp : PyObject {
    PyObject *fieldname;
    PyObject *cb;

    composite_rtprop_data *get_data(PyObject *obj)
    {
        if (fieldname == Py_None)
            py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
        if (py_object val(PyObject_GetAttr(obj, fieldname)); !val) {
            PyErr_Clear();
        }
        else if (Py_TYPE(val.get()) == &composite_rtprop_data::Type) {
            return (composite_rtprop_data*)val.release();
        }
        py_object o(pytype_genericalloc(&composite_rtprop_data::Type));
        auto data = (composite_rtprop_data*)o.get();
        data->ovr = py_immref(Py_None);
        data->cache = py_immref(Py_None);
        throw_if(PyObject_SetAttr(obj, fieldname, o.get()));
        return (composite_rtprop_data*)o.release();
    }

    PyObject *get_res(PyObject *obj)
    {
        auto data = get_data(obj);
        py_object py_data(data);
        if (!data->filled || (!data->compiled && _object_compiled(obj))) {
            auto res = throw_if_not(PyObject_Vectorcall(cb, &obj, 1, nullptr));
            Py_DECREF(data->cache);
            data->cache = res;
            data->filled = true;
            data->compiled = _object_compiled(obj);
        }
        if (data->ovr == Py_None)
            return py_newref(data->cache);
        return apply_composite_ovr(data->cache, data->ovr);
    }

    static PyObject *get_state(CompositeRTProp *self,
                               PyObject *const *args, Py_ssize_t nargs)
    {
        return py_catch_error([&] {
            py_check_num_arg("get_state", nargs, 1, 1);
            auto data = self->get_data(args[0]);
            auto res = py_newref(data->ovr);
            Py_DECREF(data);
            return res;
        });
    }
    static PyObject *set_state(CompositeRTProp *self,
                               PyObject *const *args, Py_ssize_t nargs)
    {
        return py_catch_error([&] {
            py_check_num_arg("set_state", nargs, 2, 2);
            auto data = self->get_data(args[0]);
            pyassign(data->ovr, args[1]);
            Py_DECREF(data);
            Py_RETURN_NONE;
        });
    }

    static PyObject *set_name(CompositeRTProp *self,
                              PyObject *const *args, Py_ssize_t nargs)
    {
        return py_catch_error([&] {
            py_check_num_arg("__set_name__", nargs, 2, 2);
            auto fieldname = throw_if_not(PyUnicode_Concat("__CompositeRTProp__"_py, args[1]));
            Py_DECREF(self->fieldname);
            self->fieldname = fieldname;
            Py_RETURN_NONE;
        });
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
    .tp_methods = (PyMethodDef[]){
        {"get_state", (PyCFunction)(void*)CompositeRTProp::get_state, METH_FASTCALL, 0},
        {"set_state", (PyCFunction)(void*)CompositeRTProp::set_state, METH_FASTCALL, 0},
        {"__set_name__", (PyCFunction)(void*)CompositeRTProp::set_name, METH_FASTCALL, 0},
        {0, 0, 0, 0}
    },
    .tp_descr_get = [] (PyObject *self, PyObject *obj, PyObject*) -> PyObject* {
        if (!obj) [[unlikely]]
            return py_newref(self);
        return py_catch_error([&] { return ((CompositeRTProp*)self)->get_res(obj); });
    },
    .tp_vectorcall = [] (PyObject *type, PyObject *const *args, size_t _nargs,
                         PyObject *kwnames) -> PyObject* {
        auto nargs = PyVectorcall_NARGS(_nargs);
        if (kwnames && PyTuple_GET_SIZE(kwnames))
            return PyErr_Format(PyExc_TypeError,
                                "CompositeRTProp.__init__() got an unexpected "
                                "keyword argument '%U'", PyTuple_GET_ITEM(kwnames, 0));
        return py_catch_error([&] {
            py_check_num_arg("CompositeRTProp.__init__", nargs, 1, 1);
            auto cb = args[0];
            py_object o(pytype_genericalloc(&CompositeRTProp_Type));
            auto data = (CompositeRTProp*)o.get();
            data->fieldname = py_immref(Py_None);
            data->cb = py_newref(cb);
            return o.release();
        });
    },
};

namespace {

#define RTPROP_PREFIX_STR "_RTProp_value_"
static constexpr int rtprop_prefix_len = strlen(RTPROP_PREFIX_STR);

struct rtprop_callback : ExternCallback {
    PyObject *obj;
    PyObject *fieldname;

    static TagVal callback(rtprop_callback *self, unsigned age)
    {
        auto v = pyobj_checked(PyObject_GetAttr(self->obj, self->fieldname));
        if (!is_rtval(v))
            return TagVal::from_py(v);
        auto rv = (RuntimeValue*)v.get();
        if (rv->type_ == ExternAge && rv->cb_arg2 == self)
            py_throw_format(PyExc_ValueError, "RT property have not been assigned.");
        rt_eval_cache(rv, age);
        return rtval_cache(rv);
    }

    static inline rtprop_callback *alloc(PyObject *obj, PyObject *fieldname)
    {
        auto self = (rtprop_callback*)pytype_genericalloc(&Type);
        self->fptr = (void*)callback;
        self->obj = py_newref(obj);
        self->fieldname = py_newref(fieldname);
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
    .tp_str = [] (PyObject *py_self) {
        return py_catch_error([&] {
            auto self = (rtprop_callback*)py_self;
            return PyUnicode_FromFormat(
                "<RTProp %U for %S>",
                pyobj_checked(PyUnicode_Substring(self->fieldname, rtprop_prefix_len,
                                                  PY_SSIZE_T_MAX)).get(), self->obj);
        });
    },
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

    PyObject *get_res(PyObject *obj)
    {
        if (fieldname == Py_None)
            py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
        if (auto res = PyObject_GetAttr(obj, fieldname))
            return res;
        PyErr_Clear();
        py_object val(new_extern_age(
                          py_object(rtprop_callback::alloc(obj, fieldname)).get(),
                          (PyObject*)&PyFloat_Type));
        throw_if(PyObject_SetAttr(obj, fieldname, val));
        return val.release();
    }

    void set_res(PyObject *obj, PyObject *val)
    {
        if (fieldname == Py_None)
            py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
        if (val && val != Py_None)
            throw_if(PyObject_SetAttr(obj, fieldname, val));
        else
            throw_if(PyObject_DelAttr(obj, fieldname));
    }

    static PyObject *get_state(RTProp *self, PyObject *const *args, Py_ssize_t nargs)
    {
        return py_catch_error([&] {
            py_check_num_arg("get_state", nargs, 1, 1);
            if (auto res = PyObject_GetAttr(args[0], self->fieldname))
                return res;
            PyErr_Clear();
            Py_RETURN_NONE;
        });
    }

    static PyObject *set_state(RTProp *self, PyObject *const *args, Py_ssize_t nargs)
    {
        return py_catch_error([&] {
            py_check_num_arg("set_state", nargs, 2, 2);
            auto obj = args[0];
            auto val = args[1];
            if (val == Py_None)
                throw_if(PyObject_DelAttr(obj, self->fieldname));
            else
                throw_if(PyObject_SetAttr(obj, self->fieldname, val));
            Py_RETURN_NONE;
        });
    }

    static PyObject *set_name(RTProp *self, PyObject *const *args, Py_ssize_t nargs)
    {
        return py_catch_error([&] {
            py_check_num_arg("__set_name__", nargs, 2, 2);
            auto fieldname = throw_if_not(PyUnicode_Concat(RTPROP_PREFIX_STR ""_py, args[1]));
            Py_DECREF(self->fieldname);
            self->fieldname = fieldname;
            Py_RETURN_NONE;
        });
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
    .tp_methods = (PyMethodDef[]){
        {"get_state", (PyCFunction)(void*)RTProp::get_state, METH_FASTCALL, 0},
        {"set_state", (PyCFunction)(void*)RTProp::set_state, METH_FASTCALL, 0},
        {"__set_name__", (PyCFunction)(void*)RTProp::set_name, METH_FASTCALL, 0},
        {0, 0, 0, 0}
    },
    .tp_descr_get = [] (PyObject *py_self, PyObject *obj, PyObject*) {
        if (!obj) [[unlikely]]
            return py_newref(py_self);
        return py_catch_error([&] { return ((RTProp*)py_self)->get_res(obj); });
    },
    .tp_descr_set = [] (PyObject *py_self, PyObject *obj, PyObject *val) {
        try {
            ((RTProp*)py_self)->set_res(obj, val);
        }
        catch (...) {
            catch_cxx_error();
            return -1;
        }
        return 0;
    },
    .tp_vectorcall = [] (PyObject *type, PyObject *const *args, size_t _nargs,
                         PyObject *kwnames) -> PyObject* {
        auto nargs = PyVectorcall_NARGS(_nargs);
        if (kwnames && PyTuple_GET_SIZE(kwnames))
            return PyErr_Format(PyExc_TypeError,
                                "RTProp.__init__() got an unexpected "
                                "keyword argument '%U'", PyTuple_GET_ITEM(kwnames, 0));
        return py_catch_error([&] {
            py_check_num_arg("RTProp.__init__", nargs, 0, 0);
            auto self = (RTProp*)pytype_genericalloc(&RTProp_Type);
            self->fieldname = py_immref(Py_None);
            return self;
        });
    },
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
