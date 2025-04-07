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

struct composite_rtprop_data {
    PyObject_HEAD
    PyObject *ovr;
    PyObject *cache;
    bool compiled;
    bool filled;
};
static PyTypeObject composite_rtprop_data_Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.rtval.composite_rtprop_data",
    .tp_basicsize = sizeof(composite_rtprop_data),
    .tp_dealloc = [] (PyObject *py_self) {
        PyObject_GC_UnTrack(py_self);
        composite_rtprop_data_Type.tp_clear(py_self);
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

static inline __attribute__((returns_nonnull)) composite_rtprop_data*
get_composite_rtprop_data(CompositeRTProp *prop, PyObject *obj)
{
    auto fieldname = prop->fieldname;
    if (fieldname == Py_None)
        py_throw_format(PyExc_ValueError, "Cannot determine runtime property name");
    if (py_object val(PyObject_GetAttr(obj, fieldname)); !val) {
        PyErr_Clear();
    }
    else if (Py_TYPE(val.get()) == &composite_rtprop_data_Type) {
        return (composite_rtprop_data*)val.release();
    }
    py_object o(pytype_genericalloc(&composite_rtprop_data_Type));
    auto data = (composite_rtprop_data*)o.get();
    data->ovr = py_immref(Py_None);
    data->cache = py_immref(Py_None);
    throw_if(PyObject_SetAttr(obj, fieldname, o.get()));
    return (composite_rtprop_data*)o.release();
}

static inline __attribute__((returns_nonnull)) PyObject*
apply_composite_ovr(PyObject *val, PyObject *ovr);

static inline bool
apply_dict_ovr(PyObject *dict, PyObject *k, PyObject *v)
{
    auto field = PyDict_GetItemWithError(dict, k);
    if (field) {
        py_object newfield(apply_composite_ovr(field, v));
        throw_if(PyDict_SetItem(dict, k, newfield));
        return true;
    }
    throw_if(PyErr_Occurred());
    return false;
}

static inline __attribute__((returns_nonnull)) PyObject*
apply_composite_ovr(PyObject *val, PyObject *ovr)
{
    if (!PyDict_Check(ovr))
        return py_newref(ovr);
    if (!PyDict_Size(ovr))
        return py_newref(val);
    if (PyDict_Check(val)) {
        py_object newval(throw_if_not(PyDict_Copy(val)));
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
        py_object newval(throw_if_not(PySequence_List(val)));
        for (auto [k, v]: pydict_iter(ovr)) {
            // for scangroup support since only string key is supported
            py_object ik(throw_if_not(PyNumber_Long(k)));
            auto idx = PyLong_AsLong(ik.get());
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

static inline __attribute__((returns_nonnull)) PyObject*
composite_rtprop_get_res(CompositeRTProp *self, PyObject *obj)
{
    auto data = get_composite_rtprop_data(self, obj);
    py_object py_data((PyObject*)data);
    if (!data->filled || (!data->compiled && _object_compiled(obj))) {
        py_object res(throw_if_not(PyObject_Vectorcall(self->cb, &obj, 1, nullptr)));
        Py_DECREF(data->cache);
        data->cache = res.release();
        data->filled = true;
        data->compiled = _object_compiled(obj);
    }
    if (data->ovr == Py_None)
        return py_newref(data->cache);
    return apply_composite_ovr(data->cache, data->ovr);
}

static PyObject *composite_rtprop_get_state(PyObject *py_self, PyObject *const *args,
                                            Py_ssize_t nargs)
{
    return py_catch_error([&] {
        auto self = (CompositeRTProp*)py_self;
        py_check_num_arg("get_state", nargs, 1, 1);
        auto data = get_composite_rtprop_data(self, args[0]);
        auto res = py_newref(data->ovr);
        Py_DECREF(data);
        return res;
    });
}

static PyObject *composite_rtprop_set_state(PyObject *py_self, PyObject *const *args,
                                            Py_ssize_t nargs)
{
    return py_catch_error([&] {
        auto self = (CompositeRTProp*)py_self;
        py_check_num_arg("set_state", nargs, 2, 2);
        auto data = get_composite_rtprop_data(self, args[0]);
        pyassign(data->ovr, args[1]);
        Py_DECREF(data);
        Py_RETURN_NONE;
    });
}

static PyObject *composite_rtprop_set_name(PyObject *py_self, PyObject *const *args,
                                           Py_ssize_t nargs)
{
    return py_catch_error([&] {
        auto self = (CompositeRTProp*)py_self;
        py_check_num_arg("__set_name__", nargs, 2, 2);
        auto fieldname = throw_if_not(PyUnicode_Concat("__CompositeRTProp__"_py, args[1]));
        Py_DECREF(self->fieldname);
        self->fieldname = fieldname;
        Py_RETURN_NONE;
    });
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
        {"get_state", (PyCFunction)(void*)composite_rtprop_get_state, METH_FASTCALL, 0},
        {"set_state", (PyCFunction)(void*)composite_rtprop_set_state, METH_FASTCALL, 0},
        {"__set_name__", (PyCFunction)(void*)composite_rtprop_set_name, METH_FASTCALL, 0},
        {0, 0, 0, 0}
    },
    .tp_descr_get = [] (PyObject *self, PyObject *obj, PyObject*) -> PyObject* {
        if (!obj) [[unlikely]]
            return py_newref(self);
        return py_catch_error([&] {
            return composite_rtprop_get_res((CompositeRTProp*)self, obj); });
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

__attribute__((visibility("protected")))
void init()
{
    throw_if(PyType_Ready(&composite_rtprop_data_Type) < 0);
    throw_if(PyType_Ready(&CompositeRTProp_Type) < 0);
}

}
