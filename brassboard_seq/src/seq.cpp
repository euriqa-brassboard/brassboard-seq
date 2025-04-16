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

static PyObject *timestep_type;
static PyObject *subseq_type;
static PyObject *condwrapper_type;

using namespace rtval;
using event_time::EventTime;

static inline int get_channel_id(auto self, PyObject *name)
{
    auto channel_name_map = py::dict(self->channel_name_map);
    if (auto chn = channel_name_map.try_get(name)) [[likely]]
        return PyLong_AsLong(chn);
    auto path = py::tuple_ref(self->config->translate_channel(name));
    auto channel_path_map = py::dict(self->channel_path_map);
    if (auto chn = channel_path_map.try_get(path)) {
        channel_name_map.set(name, chn);
        return PyLong_AsLong(chn);
    }
    auto channel_paths = py::list(self->channel_paths);
    int cid = channel_paths.size();
    channel_paths.append(path);
    auto pycid = py::new_int(cid);
    channel_path_map.set(path, pycid);
    channel_name_map.set(name, pycid);
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
    auto cond2 = rt_convert_bool(new_cond);
    if (cond1 == Py_True)
        return { cond2.rel(), true };
    assert(is_rtval(cond1));
    auto self = py::generic_alloc<RuntimeValue>();
    self->datatype = DataType::Bool;
    // self->cache_err = EvalError::NoError;
    // self->cache_val = { .i64_val = 0 };
    self->type_ = And;
    self->age = (unsigned)-1;
    self->arg0 = (RuntimeValue*)py::newref(cond1);
    self->arg1 = cond2.rel();
    self->cb_arg2 = py::immref(Py_None);
    return { (PyObject*)self.rel(), true };
}

static inline __attribute__((returns_nonnull)) PyObject*
combine_cond(PyObject *cond1, PyObject *new_cond)
{
    auto [res, needs_free] = _combine_cond(cond1, new_cond);
    if (!needs_free)
        return py::newref(res);
    return res;
}

static inline void timeseq_set_time(auto self, EventTime *time, PyObject *offset)
{
    if (is_rtval(offset))
        self->start_time->set_base_rt(time, event_time::round_time_rt(offset));
    else
        self->start_time->set_base_int(time, event_time::round_time_int(offset));
    self->seqinfo->bt_tracker.record(event_time_key(self->start_time));
}

template<typename TimeStep> static inline py::ref<TimeStep>
add_time_step(auto self, py::ptr<> cond, py::ptr<EventTime> start_time, py::ptr<> length)
{
    auto seqinfo = pyx_fld(self, seqinfo);
    py::ref end_time(seqinfo->time_mgr->new_round(start_time, length,
                                                  cond, (EventTime*)Py_None));
    auto step = py::generic_alloc<TimeStep>(timestep_type);
    call_constructor(&step->actions);
    pyx_fld(step, seqinfo) = py::newref(seqinfo);
    pyx_fld(step, start_time) = py::newref(start_time);
    pyx_fld(step, end_time) = end_time.rel();
    pyx_fld(step, cond) = py::newref(cond);
    pyx_fld(step, length) = py::newref(length);
    seqinfo->bt_tracker.record(event_time_key(pyx_fld(step, end_time)));
    py::list(self->sub_seqs).append(step);
    return step;
}

template<typename SubSeq> static inline py::ref<SubSeq>
add_custom_step(SubSeq *self, py::ptr<> cond, py::ptr<EventTime> start_time,
                py::ptr<> cb, size_t nargs, PyObject *const *args, py::tuple kwnames)
{
    auto seqinfo = pyx_fld(self, seqinfo);
    auto subseq = py::generic_alloc<SubSeq>(subseq_type);
    pyx_fld(subseq, seqinfo) = py::newref(seqinfo);
    pyx_fld(subseq, start_time) = py::newref(start_time);
    pyx_fld(subseq, end_time) = py::newref(start_time);
    pyx_fld(subseq, cond) = py::newref(cond);
    pyx_fld(subseq, sub_seqs) = py::new_list(0).rel();
    subseq->dummy_step = (decltype(subseq->dummy_step))py::immref(Py_None);
    // The python vectorcall ABI allows us to temporarily change the argument array
    // as long as we restore it before returning.
    auto prev_arg = args[-1];
    ((PyObject**)args)[-1] = (PyObject*)subseq;
    ScopeExit restore_arg([&] { ((PyObject**)args)[-1] = prev_arg; });
    cb.vcall(&args[-1], nargs + 1, kwnames);
    py::list(self->sub_seqs).append(subseq);
    return subseq;
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
        return py::newref(cond);
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

template<typename CondSeq, typename TimeSeq, typename TimeStep, AddStepType type>
static PyObject *add_step_real(PyObject *py_self, PyObject *const *args,
                               Py_ssize_t nargs, PyObject *_kwnames) try
{
    auto self = (CondSeq*)py_self;
    auto subseq = condseq_get_subseq(self);
    auto cond = pyx_fld(self, cond);
    auto nargs_min = type == AddStepType::At ? 2 : 1;
    py_check_num_arg(add_step_name(type), nargs, nargs_min);

    auto first_arg = args[nargs_min - 1];
    py::ref<EventTime> start_time;
    if (type == AddStepType::Background) {
        start_time.assign(pyx_fld(subseq, end_time));
    }
    else if (type == AddStepType::Floating) {
        auto time_mgr = pyx_fld(subseq, seqinfo)->time_mgr;
        start_time.take(time_mgr->new_int((EventTime*)Py_None, 0,
                                          true, cond, (EventTime*)Py_None));
    }
    else if (type == AddStepType::At) {
        if (args[0] != Py_None && !py::typeis<event_time::EventTime>(args[0]))
            return PyErr_Format(PyExc_TypeError,
                                "Argument 'tp' has incorrect type (expected EventTime, "
                                "got %.200s)", Py_TYPE(args[0])->tp_name);
        start_time.assign(args[0]);
    }
    else {
        assert(type == AddStepType::Step);
        start_time.assign(pyx_fld(subseq, end_time));
    }

    auto tuple_nargs = nargs - nargs_min;
    auto get_args_tuple = [&] {
        if (tuple_nargs == 0)
            return py::empty_tuple.immref();
        auto res = py::new_tuple(tuple_nargs);
        auto *tuple_args = args + nargs_min;
        for (auto i = 0; i < tuple_nargs; i++)
            res.SET(i, py::ptr(tuple_args[i]));
        return res;
    };

    py::ref<TimeSeq> res;
    if (Py_TYPE(first_arg)->tp_call) {
        assert(nargs_min >= 1);
        res.take(add_custom_step(subseq, cond, start_time, first_arg,
                                 tuple_nargs, args + nargs_min, _kwnames));
    }
    else if (auto kwnames = py::tuple(_kwnames); kwnames && kwnames.size()) {
        auto kws = py::new_dict();
        auto kwvalues = args + nargs;
        for (auto [i, name]: py::tuple_iter(kwnames))
            kws.set(name, kwvalues[i]);
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S, %S.",
                            get_args_tuple(), kws);
    }
    else if (tuple_nargs == 0) {
        res.take(add_time_step<TimeStep>(subseq, cond, start_time, first_arg));
    }
    else {
        return PyErr_Format(PyExc_ValueError,
                            "Unexpected arguments when creating new time step, %S.",
                            get_args_tuple());
    }
    if (type == AddStepType::Step)
        py::assign(pyx_fld(subseq, end_time), res->end_time);
    return (PyObject*)res.rel();
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

static inline void
timestep_set(auto *self, PyObject *chn, PyObject *value, PyObject *cond,
             bool is_pulse, bool exact_time, py::dict_ref &&kws)
{
    auto seqinfo = pyx_fld(self, seqinfo);
    int cid;
    if (py::typeis<py::int_>(chn)) {
        auto lcid = PyLong_AsLong(chn);
        if (lcid < 0 || lcid > py::list(seqinfo->channel_paths).size())
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
        py_throw_format(PyExc_ValueError,
                        "Multiple actions added for the same channel "
                        "at the same time on %U.", channel_name_from_id(seqinfo, cid));
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
           bool exact_time, py::dict_ref &&kws)
{
    auto *step = self->dummy_step;
    using TimeStep = std::remove_reference_t<decltype(*step)>;
    auto *start_time = pyx_fld(self, end_time);
    if ((PyObject*)step == Py_None || pyx_fld(step, end_time) != start_time) {
        auto new_step = add_time_step<TimeStep>(self, pyx_fld(self, cond),
                                                start_time, py::int_cached(0));
        step = new_step.get();
        // Steals a reference while keeping step as a borrowed reference.
        py::assign(self->dummy_step, std::move(new_step));
        // Update the current time so that a normal step added later
        // this is treated as ordered after this set event
        // rather than at the same time.
        py::assign(pyx_fld(self, end_time), pyx_fld(step, end_time));
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
    py::dict_ref kws;
    if (kwnames) {
        auto kwvalues = args + nargs;
        for (auto [i, name]: py::tuple_iter(kwnames)) {
            auto value = kwvalues[i];
            if (PyUnicode_CompareWithASCIIString(name, "cond") == 0) {
                arg_cond = value;
            }
            else if (PyUnicode_CompareWithASCIIString(name, "exact_time") == 0) {
                exact_time = get_value_bool(value, (uintptr_t)-1);
            }
            else {
                if (!kws)
                    kws = py::new_dict();
                kws.set(name, value);
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
    return py::newref(py_self);
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

template<typename ConditionalWrapper> static PyObject*
condwrapper_vectorcall(ConditionalWrapper *self, PyObject *const *args, size_t _nargs,
                       PyObject *kwnames) try {
    auto nargs = PyVectorcall_NARGS(_nargs);
    py_check_no_kwnames("__call__", kwnames);
    py_check_num_arg("__call__", nargs, 1, 1);
    // Reuse the args buffer
    auto step = add_custom_step(self->seq, self->cond, pyx_fld(self->seq, end_time),
                                args[0], 0, &args[1], py::tuple());
    py::assign(pyx_fld(self->seq, end_time), pyx_fld(step, end_time));
    return (PyObject*)step.rel();
}
catch (...) {
    handle_cxx_exception();
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
    auto wrapper = py::generic_alloc<ConditionalWrapper>(condwrapper_type);
    wrapper->seq = py::newref(subseq);
    wrapper->cond = cc.take_cond();
    wrapper->fptr = (void*)condwrapper_vectorcall<ConditionalWrapper>;
    return (PyObject*)wrapper.rel();
}
catch (...) {
    handle_cxx_exception();
    return nullptr;
}

template<typename TimeStep>
static inline void update_timestep(TimeStep*)
{
    static PyMethodDef methods[] = {
        {"set", (PyCFunction)(void*)condseq_set<TimeStep,true,false>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"pulse", (PyCFunction)(void*)condseq_set<TimeStep,true,true>,
         METH_FASTCALL|METH_KEYWORDS, 0},
    };
    for (auto &method: methods)
        pytype_add_method(timestep_type, &method);
    PyType_Modified((PyTypeObject*)timestep_type);
}

template<typename SubSeq, typename ConditionalWrapper, typename TimeSeq,
         typename TimeStep>
static inline void update_subseq(SubSeq*, ConditionalWrapper*, TimeSeq*, TimeStep*)
{
    static PyMethodDef methods[] = {
        {"conditional", (PyCFunction)(void*)condseq_conditional<SubSeq,ConditionalWrapper>,
         METH_FASTCALL, 0},
        {"set", (PyCFunction)(void*)condseq_set<SubSeq>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_step",
         (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,AddStepType::Step>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_background",
         (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,AddStepType::Background>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_floating",
         (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,AddStepType::Floating>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_at",
         (PyCFunction)(void*)add_step_real<SubSeq,TimeSeq,TimeStep,AddStepType::At>,
         METH_FASTCALL|METH_KEYWORDS, 0},
    };
    for (auto &method: methods)
        pytype_add_method(subseq_type, &method);
    PyType_Modified((PyTypeObject*)subseq_type);
}

template<typename ConditionalWrapper, typename TimeSeq, typename TimeStep>
static inline void
update_conditional(ConditionalWrapper*, TimeSeq*, TimeStep*)
{
    static PyMethodDef methods[] = {
        {"conditional",
         (PyCFunction)(void*)condseq_conditional<ConditionalWrapper,ConditionalWrapper>,
         METH_FASTCALL, 0},
        {"set", (PyCFunction)(void*)condseq_set<ConditionalWrapper>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_step",
         (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,AddStepType::Step>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_background",
         (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,AddStepType::Background>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_floating",
         (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,AddStepType::Floating>,
         METH_FASTCALL|METH_KEYWORDS, 0},
        {"add_at",
         (PyCFunction)(void*)add_step_real<ConditionalWrapper,TimeSeq,TimeStep,AddStepType::At>,
         METH_FASTCALL|METH_KEYWORDS, 0},
    };
    for (auto &method: methods)
        pytype_add_method(condwrapper_type, &method);
    ((PyTypeObject*)condwrapper_type)->tp_call = PyVectorcall_Call;
    ((PyTypeObject*)condwrapper_type)->tp_vectorcall_offset =
        offsetof(ConditionalWrapper, fptr),
    PyType_Modified((PyTypeObject*)condwrapper_type);
}

}
