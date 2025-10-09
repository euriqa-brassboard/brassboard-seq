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

#include "artiq_utils.h"

#include "rtval.h"

namespace brassboard_seq::artiq_utils {

namespace {

static rtval::TagVal evalonce_callback(auto *self)
{
    if (!self->value)
        py_throw_format(PyExc_RuntimeError, "Value evaluated too early");
    return rtval::TagVal::from_py(self->value);
}

struct EvalOnceCallback : rtval::ExternCallback {
    py::ref<> value;
    py::ref<> callback;

    static inline py::ref<EvalOnceCallback> alloc(py::ptr<> callback)
    {
        auto self = py::generic_alloc<EvalOnceCallback>();
        self->fptr = (void*)evalonce_callback<EvalOnceCallback>;
        call_constructor(&self->value);
        call_constructor(&self->callback, py::newref(callback));
        return self;
    }
    static PyTypeObject Type;
};

PyTypeObject EvalOnceCallback::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.artiq_backend.EvalOnceCallback",
    .tp_basicsize = sizeof(EvalOnceCallback),
    .tp_dealloc = py::tp_cxx_dealloc<true,EvalOnceCallback>,
    .tp_str = py::unifunc<[] (py::ptr<EvalOnceCallback> self) {
        return py::str_format("(%S)()", self->callback);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<EvalOnceCallback,&EvalOnceCallback::value,&EvalOnceCallback::callback>,
    .tp_clear = py::tp_field_clear<EvalOnceCallback,&EvalOnceCallback::value,&EvalOnceCallback::callback>,
    .tp_base = &ExternCallback::Type,
};

struct DatasetCallback : rtval::ExternCallback {
    py::ref<> value;
    py::ref<> cb;
    py::str_ref key;
    py::ref<> def_val;

    static inline py::ref<DatasetCallback> alloc(py::ptr<> cb, py::str key,
                                                 py::ptr<> def_val)
    {
        auto self = py::generic_alloc<DatasetCallback>();
        self->fptr = (void*)evalonce_callback<DatasetCallback>;
        call_constructor(&self->value);
        call_constructor(&self->cb, py::newref(cb));
        call_constructor(&self->key, py::newref(key));
        call_constructor(&self->def_val, def_val.xref());
        return self;
    }
    static PyTypeObject Type;
};

PyTypeObject DatasetCallback::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.artiq_backend.DatasetCallback",
    .tp_basicsize = sizeof(DatasetCallback),
    .tp_dealloc = py::tp_cxx_dealloc<true,DatasetCallback>,
    .tp_str = py::unifunc<[] (py::ptr<DatasetCallback> self) {
        if (!PyMethod_Check(self->cb))
            return py::str_format("<dataset %S for %S>", self->key, self->cb);
        py::ptr func = PyMethod_GET_FUNCTION((PyObject*)self->cb);
        py::ptr obj = PyMethod_GET_SELF((PyObject*)self->cb);
        if (py::arg_cast<py::str>(func.attr("__name__"_py),
                                  "__name__").compare_ascii("get_dataset_sys") == 0)
            return py::str_format("<dataset_sys %S for %S>", self->key, obj);
        return py::str_format("<dataset %S for %S>", self->key, obj);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC,
    .tp_traverse = py::tp_field_traverse<DatasetCallback,&DatasetCallback::value,&DatasetCallback::cb,&DatasetCallback::def_val>,
    .tp_clear = py::tp_field_clear<DatasetCallback,&DatasetCallback::value,&DatasetCallback::cb,&DatasetCallback::key,&DatasetCallback::def_val>,
    .tp_base = &ExternCallback::Type,
};

struct SeqVariable : rtval::ExternCallback {
    rtval::TagVal value;

    static rtval::TagVal callback(SeqVariable *self)
    {
        return self->value;
    }

    static inline py::ref<SeqVariable> alloc(rtval::TagVal init)
    {
        auto self = py::generic_alloc<SeqVariable>();
        self->fptr = (void*)callback;
        call_constructor(&self->value, init);
        return self;
    }
    static PyTypeObject Type;
};

PyTypeObject SeqVariable::Type = {
    .ob_base = PyVarObject_HEAD_INIT(0, 0)
    .tp_name = "brassboard_seq.artiq_backend.SeqVariable",
    .tp_basicsize = sizeof(SeqVariable),
    .tp_dealloc = py::tp_cxx_dealloc<false,SeqVariable>,
    .tp_str = py::unifunc<[] (py::ptr<SeqVariable> self) {
        return py::str_format("var(%S)", self->value.to_py());
    }>,
    .tp_getattro = py::binfunc<[] (py::ptr<SeqVariable> self,
                                   py::str name) -> py::ref<> {
        if (name.compare_ascii("value") == 0)
            return self->value.to_py();
        return py::ref(PyObject_GenericGetAttr(self, name));
    }>,
    .tp_setattro = py::itrifunc<[] (py::ptr<SeqVariable> self, py::str name,
                                    py::ptr<> value) {
        if (name.compare_ascii("value") == 0) {
            self->value = rtval::TagVal::from_py(value);
            return 0;
        }
        return PyObject_GenericSetAttr(self, name, value);
    }>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_clear = py::tp_field_clear<SeqVariable>,
    .tp_base = &ExternCallback::Type,
};

static inline py::ref<> check_bb_rt_values(py::ptr<> self)
{
    auto vals = self.try_attr("_bb_rt_values"_py);
    if (vals)
        return vals;
    vals.take(py::new_dict());
    self.set_attr("_bb_rt_values"_py, vals);
    return vals;
}

static inline py::ref<> _call_with_opt(py::ptr<> cb, py::ptr<> key, py::ptr<> def_val)
{
    if (def_val) {
        return cb(key, def_val);
    }
    else {
        return cb(key);
    }
}

template<bool sys>
static inline py::ref<> _rt_dataset(py::ptr<> self, PyObject *const *args,
                                    Py_ssize_t nargs, py::tuple kwnames)
{
    auto fname = sys ? "rt_dataset_sys" : "rt_dataset";
    py::check_num_arg(fname, nargs, 1, 2);
    auto key = py::arg_cast<py::str>(args[0], "key");
    auto [def_val] = py::parse_pos_or_kw_args<"default">(fname, args + 1,
                                                         nargs - 1, kwnames);
    auto _vals = check_bb_rt_values(self);
    auto cb = self.attr(sys ? "get_dataset_sys"_py : "get_dataset"_py);
    if (_vals.is_none())
        return _call_with_opt(cb, key, def_val);
    auto vals = py::arg_cast<py::dict>(_vals, "_bb_rt_values");
    auto _key = py::new_tuple(key, to_py(sys));
    auto res = vals.try_get(_key);
    if (res)
        return rtval::new_extern(std::move(res), &PyFloat_Type);
    auto rtcb = DatasetCallback::alloc(cb, key, def_val);
    vals.set(_key, rtcb);
    return rtval::new_extern(std::move(rtcb), &PyFloat_Type);
}

static PyMethodDef env_methods[] = {
    py::meth_noargs<"_eval_all_rtvals", [] (py::ptr<> self) {
        auto vals = self.try_attr("_bb_rt_values"_py);
        if (vals.is_none())
            return;
        if (vals && vals.isa<py::dict>()) {
            for (auto [_, val]: py::dict_iter(vals)) {
                if (auto dval = py::cast<DatasetCallback>(val)) {
                    dval->value.take(_call_with_opt(dval->cb, dval->key, dval->def_val));
                }
                else if (auto eoval = py::cast<EvalOnceCallback>(val)) {
                    eoval->value.take(eoval->callback());
                }
                else {
                    py_throw_format(PyExc_RuntimeError,
                                    "Unknown object in runtime callbacks");
                }
            }
        }
        self.set_attr("_bb_rt_values"_py, py::new_none());
        self.attr("call_child_method"_py)("_eval_all_rtvals"_py);
    }>,
    py::meth_o<"rt_value", [] (py::ptr<> self, py::ptr<> cb) -> py::ref<> {
        auto vals = check_bb_rt_values(self);
        if (vals.is_none())
            return cb();
        auto rtcb = EvalOnceCallback::alloc(cb);
        py::arg_cast<py::dict>(vals, "_bb_rt_values").set(cb, rtcb);
        return rtval::new_extern(std::move(rtcb), &PyFloat_Type);
    }>,
    py::meth_fast<"seq_variable",[] (auto, PyObject *const *args, Py_ssize_t nargs) {
        py::check_num_arg("seq_variable", nargs, 0, 1);
        rtval::TagVal init(0.0);
        if (nargs >= 1)
            init = rtval::TagVal::from_py(args[0]);
        auto rtcb = SeqVariable::alloc(init);
        return rtval::new_extern(std::move(rtcb), &PyFloat_Type);
    }>,
    py::meth_fastkw<"rt_dataset",_rt_dataset<false>>,
    py::meth_fastkw<"rt_dataset_sys",_rt_dataset<true>>
};

} // (anonymous)

void patch_artiq()
{
    for (auto &def: env_methods) {
        py::imp<"artiq.language.environment","HasEnvironment">().set_attr(
            def.ml_name, py::ref<>::checked(PyDescr_NewMethod(&PyBaseObject_Type, &def)));
    }
}

__attribute__((visibility("hidden")))
void init()
{
    throw_if(PyType_Ready(&EvalOnceCallback::Type) < 0);
    throw_if(PyType_Ready(&DatasetCallback::Type) < 0);
    throw_if(PyType_Ready(&SeqVariable::Type) < 0);
}

}
