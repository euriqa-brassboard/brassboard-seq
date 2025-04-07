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

#include "seq.h"

#include "Python.h"

#include "event_time.h"

#include <algorithm>
#include <vector>

namespace brassboard_seq::seq {

static PyObject *event_time_type;
static PyObject *timestep_type;
static PyObject *subseq_type;
static PyObject *condwrapper_type;
static PyObject *rt_time_scale;

using namespace rtval;

static inline int get_channel_id(auto self, PyObject *name)
{
    auto channel_name_map = self->channel_name_map;
    if (auto chn = PyDict_GetItemWithError(channel_name_map, name)) [[likely]]
        return PyLong_AsLong(chn);
    throw_if(PyErr_Occurred());
    py_object path(config::translate_channel(self->config, name));
    auto channel_path_map = self->channel_path_map;
    if (auto chn = PyDict_GetItemWithError(channel_path_map, path)) {
        throw_if(PyDict_SetItem(channel_name_map, name, chn));
        return PyLong_AsLong(chn);
    }
    throw_if(PyErr_Occurred());
    auto channel_paths = self->channel_paths;
    int cid = PyList_GET_SIZE(channel_paths);
    pylist_append(channel_paths, path);
    py_object pycid(pylong_from_long(cid));
    throw_if(PyDict_SetItem(channel_path_map, path, pycid));
    throw_if(PyDict_SetItem(channel_name_map, name, pycid));
    return cid;
}

static inline std::pair<PyObject*,bool>
_combine_cond(PyObject *cond1, PyObject *new_cond)
{
    if (cond1 == Py_False)
        return { Py_False, false };
    if (!is_rtval(new_cond)) {
        if (get_value_bool(new_cond, (uintptr_t)-1)) {
            return { cond1, false };
        }
        else {
            return { Py_False, false };
        }
    }
    py_object cond2((PyObject*)rt_convert_bool((RuntimeValue*)new_cond));
    if (cond1 == Py_True)
        return { cond2.release(), true };
    assert(is_rtval(cond1));
    auto o = pytype_genericalloc(&RuntimeValue_Type);
    auto self = (RuntimeValue*)o;
    self->datatype = DataType::Bool;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = And;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py_newref(cond1);
    self->arg1 = (RuntimeValue*)cond2.release();
    self->cb_arg2 = py_immref(Py_None);
    return { o, true };
}

static inline __attribute__((returns_nonnull)) PyObject*
combine_cond(PyObject *cond1, PyObject *new_cond)
{
    auto [res, needs_free] = _combine_cond(cond1, new_cond);
    if (!needs_free)
        Py_INCREF(res);
    return res;
}

template<typename EventTime>
static inline __attribute__((returns_nonnull)) EventTime*
new_round_time(auto self, EventTime *prev, PyObject *offset, PyObject *cond,
               EventTime *wait_for)
{
    if (is_rtval(offset)) {
        py_object rt_offset((PyObject*)event_time::round_time_rt(
                                (RuntimeValue*)offset, (RuntimeValue*)rt_time_scale));
        return event_time::_new_time_rt(self, event_time_type, prev,
                                        (RuntimeValue*)rt_offset.get(),
                                        cond, wait_for);
    }
    else {
        auto coffset = event_time::round_time_int(offset);
        return event_time::_new_time_int(self, event_time_type, prev,
                                         coffset, false, cond, wait_for);
    }
}

template<typename TimeStep, typename EventTime>
static inline __attribute__((returns_nonnull)) TimeStep*
add_time_step(auto self, PyObject *cond, EventTime *start_time, PyObject *length)
{
    auto seqinfo = pyx_fld(self, seqinfo);
    py_object end_time((PyObject*)new_round_time(seqinfo->time_mgr, start_time, length,
                                                 cond, (EventTime*)Py_None));
    py_object o(pytype_genericalloc(timestep_type));
    auto step = (TimeStep*)o.get();
    new (&step->actions) std::vector<py_object>();
    pyx_fld(step, seqinfo) = py_newref(seqinfo);
    pyx_fld(step, start_time) = py_newref(start_time);
    pyx_fld(step, end_time) = (EventTime*)end_time.release();
    pyx_fld(step, cond) = py_newref(cond);
    pyx_fld(step, length) = py_newref(length);
    seqinfo->bt_tracker.record(event_time_key(pyx_fld(step, end_time)));
    pylist_append(self->sub_seqs, o);
    return (TimeStep*)o.release();
}

template<typename SubSeq, typename EventTime>
static inline __attribute__((returns_nonnull)) SubSeq*
add_custom_step(SubSeq *self, PyObject *cond, EventTime *start_time, PyObject *cb,
                size_t nargs=0, PyObject *const *args=nullptr, PyObject *kwargs=nullptr)
{
    py_object sub_seqs(pylist_new(0));
    auto seqinfo = pyx_fld(self, seqinfo);
    py_object o(pytype_genericalloc(subseq_type));
    auto subseq = (SubSeq*)o.get();
    pyx_fld(subseq, seqinfo) = py_newref(seqinfo);
    pyx_fld(subseq, start_time) = py_newref(start_time);
    pyx_fld(subseq, end_time) = py_newref(start_time);
    pyx_fld(subseq, cond) = py_newref(cond);
    pyx_fld(subseq, sub_seqs) = sub_seqs.release();
    subseq->dummy_step = (decltype(subseq->dummy_step))py_immref(Py_None);
    {
        PyObject *callargs[nargs + 1] = { o };
        for (auto i = 0; i < nargs; i++)
            callargs[i + 1] = args[i];
        py_object res(throw_if_not(PyObject_VectorcallDict(cb, callargs,
                                                           nargs + 1, kwargs)));
    }
    pylist_append(self->sub_seqs, o);
    return (SubSeq*)o.release();
}

struct CondCombiner {
    PyObject *cond{nullptr};
    bool needs_free{false};
    CondCombiner(PyObject *cond1, PyObject *cond2)
    {
        auto [_cond, _needs_free] = _combine_cond(cond1, cond2);
        cond = _cond;
        needs_free = _needs_free;
    }
    PyObject *take_cond()
    {
        if (needs_free) {
            needs_free = false;
            return cond;
        }
        return py_newref(cond);
    }
    ~CondCombiner()
    {
        if (needs_free) {
            Py_DECREF(cond);
        }
    }
};

static inline auto condseq_get_subseq(auto *self)
{
    if constexpr (requires { self->seq; }) {
        return self->seq;
    }
    else {
        return self;
    }
}

enum class AddStepType {
    Step,
    Background,
    Floating,
    At,
};

const char *add_step_name(AddStepType type)
{
    if (type == AddStepType::Background)
        return "add_background";
    if (type == AddStepType::Floating)
        return "add_floating";
    if (type == AddStepType::At)
        return "add_at";
    assert(type == AddStepType::Step);
    return "add_step";
}

static auto empty_tuple = PyTuple_New(0);

template<typename CondSeq, typename TimeSeq, typename TimeStep, AddStepType type>
static PyObject *add_step_real(PyObject *py_self, PyObject *const *args,
                               Py_ssize_t nargs, PyObject *kwnames) try
{
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq(self);
    auto cond = pyx_fld(self, cond);
    auto nargs_min = type == AddStepType::At ? 2 : 1;
    py_check_num_arg(add_step_name(type), nargs, nargs_min);

    auto first_arg = args[nargs_min - 1];
    using EventTime = std::remove_reference_t<decltype(*pyx_fld(subseq, end_time))>;
    py_object start_time;
    if (type == AddStepType::Background) {
        start_time.reset(py_newref((PyObject*)pyx_fld(subseq, end_time)));
    }
    else if (type == AddStepType::Floating) {
        auto time_mgr = pyx_fld(subseq, seqinfo)->time_mgr;
        auto new_time = event_time::_new_time_int(time_mgr, event_time_type,
                                                  (EventTime*)Py_None, 0, true, cond,
                                                  (EventTime*)Py_None);
        start_time.reset((PyObject*)new_time);
    }
    else if (type == AddStepType::At) {
        if (args[0] != Py_None && Py_TYPE(args[0]) != (PyTypeObject*)event_time_type)
            return PyErr_Format(PyExc_TypeError,
                                "Argument 'tp' has incorrect type (expected EventTime, "
                                "got %.200s)", Py_TYPE(args[0])->tp_name);
        start_time.reset(py_newref(args[0]));
    }
    else {
        assert(type == AddStepType::Step);
        start_time.reset(py_newref((PyObject*)pyx_fld(subseq, end_time)));
    }

    auto tuple_nargs = nargs - nargs_min;
    auto get_args_tuple = [&] {
        if (tuple_nargs == 0)
            return py_newref(empty_tuple);
        auto res = pytuple_new(tuple_nargs);
        auto *tuple_args = args + nargs_min;
        for (auto i = 0; i < tuple_nargs; i++)
            PyTuple_SET_ITEM(res, i, py_newref(tuple_args[i]));
        return res;
    };

    py_object kws;
    if (kwnames) {
        kws.reset(pydict_new());
        auto kwvalues = args + nargs;
        for (auto [i, name]: pytuple_iter(kwnames)) {
            throw_if(PyDict_SetItem(kws, name, kwvalues[i]));
        }
    }

    PyObject *res;
    if (Py_TYPE(first_arg)->tp_call) {
        res = (PyObject*)add_custom_step(subseq, cond, (EventTime*)start_time.get(),
                                         first_arg, tuple_nargs, args + nargs_min,
                                         kws.get());
    }
    else if (kws) {
        py_object arg_tuple(get_args_tuple());
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S, %S.",
                            arg_tuple.get(), kws.get());
    }
    else if (tuple_nargs == 0) {
        res = (PyObject*)add_time_step<TimeStep>(
            subseq, cond, (EventTime*)start_time.get(), first_arg);
    }
    else {
        py_object arg_tuple(get_args_tuple());
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S.",
                            arg_tuple.get());
    }
    if (type == AddStepType::Step)
        pyassign(pyx_fld(subseq, end_time), ((TimeSeq*)res)->end_time);
    return res;
}
catch (...) {
    return nullptr;
}

static inline void
timestep_set(auto *self, PyObject *chn, PyObject *value, PyObject *cond,
             bool is_pulse, bool exact_time, py_object &&kws)
{
    auto seqinfo = pyx_fld(self, seqinfo);
    int cid;
    if (Py_TYPE(chn) == &PyLong_Type) {
        auto lcid = PyLong_AsLong(chn);
        if (lcid < 0 || lcid > PyList_GET_SIZE(seqinfo->channel_paths))
            py_throw_format(PyExc_ValueError, "Channel id %ld out of bound", lcid);
        cid = lcid;
    }
    else {
        cid = get_channel_id(seqinfo, chn);
    }
    if (cid >= self->actions.size()) {
        self->actions.resize(cid + 1);
    }
    else if (self->actions[cid]) {
        auto name = channel_name_from_id(seqinfo, cid);
        py_throw_format(PyExc_ValueError,
                        "Multiple actions added for the same channel "
                        "at the same time on %U.", name.get());
    }
    auto aid = seqinfo->action_counter;
    auto action = seqinfo->action_alloc.alloc(value, cond, is_pulse, exact_time,
                                              std::move(kws), aid);
    action->length = self->length;
    seqinfo->bt_tracker.record(action_key(aid));
    seqinfo->action_counter = aid + 1;
    self->actions[cid] = action;
}

static inline void
subseq_set(auto *self, PyObject *chn, PyObject *value, PyObject *cond,
           bool exact_time, py_object &&kws)
{
    auto *step = self->dummy_step;
    using TimeStep = std::remove_reference_t<decltype(*step)>;
    auto *start_time = pyx_fld(self, end_time);
    if ((PyObject*)step == Py_None || pyx_fld(step, end_time) != start_time) {
        step = add_time_step<TimeStep>(self, pyx_fld(self, cond),
                                       start_time, pylong_cached(0));
        Py_DECREF(self->dummy_step);
        self->dummy_step = step;
        // Update the current time so that a normal step added later
        // this is treated as ordered after this set event
        // rather than at the same time.
        pyassign(pyx_fld(self, end_time), pyx_fld(step, end_time));
    }
    timestep_set(step, chn, value, cond, false, exact_time, std::move(kws));
}

template<typename CondSeq, bool is_step=false, bool is_pulse=false>
static PyObject *condseq_set(PyObject *py_self, PyObject *const *args,
                             Py_ssize_t nargs, PyObject *kwnames) try
{
    static_assert(is_step || !is_pulse);
    py_check_num_arg((is_pulse ? "pulse" : "set"), nargs, 2, 2);
    auto chn = args[0];
    auto value = args[1];
    bool exact_time{false};
    PyObject *arg_cond{Py_True};
    py_object kws;
    if (kwnames) {
        auto kwvalues = args + nargs;
        for (auto [i, name]: pytuple_iter(kwnames)) {
            auto value = kwvalues[i];
            if (PyUnicode_CompareWithASCIIString(name, "cond") == 0) {
                arg_cond = value;
            }
            else if (PyUnicode_CompareWithASCIIString(name, "exact_time") == 0) {
                exact_time = get_value_bool(value, (uintptr_t)-1);
            }
            else {
                if (!kws)
                    kws.reset(pydict_new());
                throw_if(PyDict_SetItem(kws, name, value));
            }
        }
    }
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq(self);
    auto cond = pyx_fld(self, cond);
    CondCombiner cc(cond, arg_cond);
    if constexpr (is_step)
        timestep_set(subseq, chn, value, cc.cond, is_pulse, exact_time, std::move(kws));
    else
        subseq_set(subseq, chn, value, cc.cond, exact_time, std::move(kws));
    return py_newref(py_self);
}
catch (...) {
    return nullptr;
}

template<typename CondSeq, typename ConditionalWrapper>
static PyObject *condseq_conditional(PyObject *py_self, PyObject *const *args,
                                     Py_ssize_t nargs) try
{
    py_check_num_arg("conditional", nargs, 1, 1);
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq(self);
    auto cond = pyx_fld(self, cond);
    CondCombiner cc(cond, args[0]);
    auto o = pytype_genericalloc(condwrapper_type);
    auto wrapper = (ConditionalWrapper*)o;
    wrapper->seq = py_newref(subseq);
    wrapper->cond = cc.take_cond();
    return o;
}
catch (...) {
    return nullptr;
}

template<typename TimeStep>
static inline void update_timestep(TimeStep*)
{
    static PyMethodDef timestep_set_method = {
        "set", (PyCFunction)(void*)condseq_set<TimeStep,true,false>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef timestep_pulse_method = {
        "pulse", (PyCFunction)(void*)condseq_set<TimeStep,true,true>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    pytype_add_method(timestep_type, &timestep_set_method);
    pytype_add_method(timestep_type, &timestep_pulse_method);
    PyType_Modified((PyTypeObject*)timestep_type);
}

template<typename SubSeq, typename ConditionalWrapper, typename TimeSeq,
         typename TimeStep>
static inline void update_subseq(SubSeq*, ConditionalWrapper*, TimeSeq*, TimeStep*)
{
    static PyMethodDef subseq_conditional_method = {
        "conditional", (PyCFunction)(void*)condseq_conditional<SubSeq,ConditionalWrapper>,
        METH_FASTCALL, 0};
    static PyMethodDef subseq_set_method = {
        "set", (PyCFunction)(void*)condseq_set<SubSeq>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_step_method = {
        "add_step",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,AddStepType::Step>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_background_method = {
        "add_background",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,AddStepType::Background>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_floating_method = {
        "add_floating",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,AddStepType::Floating>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef subseq_add_at_method = {
        "add_at",
        (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,AddStepType::At>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    pytype_add_method(subseq_type, &subseq_conditional_method);
    pytype_add_method(subseq_type, &subseq_set_method);
    pytype_add_method(subseq_type, &subseq_add_step_method);
    pytype_add_method(subseq_type, &subseq_add_background_method);
    pytype_add_method(subseq_type, &subseq_add_floating_method);
    pytype_add_method(subseq_type, &subseq_add_at_method);
    PyType_Modified((PyTypeObject*)subseq_type);
}

template<typename ConditionalWrapper, typename TimeSeq, typename TimeStep>
static inline void
update_conditional(ConditionalWrapper*, TimeSeq*, TimeStep*)
{
    static PyMethodDef conditional_conditional_method = {
        "conditional", (PyCFunction)(void*)condseq_conditional<ConditionalWrapper,ConditionalWrapper>,
        METH_FASTCALL, 0};
    static PyMethodDef conditional_set_method = {
        "set", (PyCFunction)(void*)condseq_set<ConditionalWrapper>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_step_method = {
        "add_step",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,AddStepType::Step>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_background_method = {
        "add_background",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,AddStepType::Background>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_floating_method = {
        "add_floating",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,AddStepType::Floating>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    static PyMethodDef conditional_add_at_method = {
        "add_at",
        (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,AddStepType::At>,
        METH_FASTCALL|METH_KEYWORDS, 0};
    pytype_add_method(condwrapper_type, &conditional_conditional_method);
    pytype_add_method(condwrapper_type, &conditional_set_method);
    pytype_add_method(condwrapper_type, &conditional_add_step_method);
    pytype_add_method(condwrapper_type, &conditional_add_background_method);
    pytype_add_method(condwrapper_type, &conditional_add_floating_method);
    pytype_add_method(condwrapper_type, &conditional_add_at_method);
    PyType_Modified((PyTypeObject*)condwrapper_type);
}

}
